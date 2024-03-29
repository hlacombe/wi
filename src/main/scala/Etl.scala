import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import better.files._

object Etl{
  def etlVectorize(df: DataFrame, action: String): (DataFrame, Int) = {

    if (File("model/pipelineETL").exists() && action == "train"){
      File("model/pipelineETL").delete()
    }

    var newDf = df
    newDf = cleanColumns(newDf,
      Array("appOrSite", "bidfloor", "city", "exchange", "impid", "interests", "label", "media",
        "network", "os", "publisher", "size", "timestamp", "type", "user"))

    newDf = splitSize(newDf)
    //newDf = splitAppOrSite(newDf)
    newDf = cleanType(newDf)
    newDf = cleanOS(newDf)
    newDf = cleanBidFloor(newDf)
    newDf = removeColumns(newDf, Array("timestamp", "impid"))
    newDf = labelToInt(newDf)
    newDf = balanceDataset(newDf)
    newDf = splitInterests(newDf)

    var vectorized = getPipelineETL(newDf).transform(newDf)

    vectorized = removeColumns(vectorized, vectorized.columns.diff(Array("features", "label", "labelIndex", "classWeightCol")))
    (vectorized, newDf.columns.length-1)
  }

  def balanceDataset(dataset: DataFrame): DataFrame = {

    // Re-balancing (weighting) of records to be used in the logistic loss objective function
    val numNegatives = dataset.filter(dataset("label") === 0).count
    val datasetSize = dataset.count
    val balancingRatio = (datasetSize - numNegatives).toDouble / datasetSize

    val calculateWeights = udf { d: Double =>
      if (d == 0.0) {
        1 * balancingRatio
      }
      else {
        (1 * (1.0 - balancingRatio))
      }
    }

    val weightedDataset = dataset.withColumn("classWeightCol", calculateWeights(dataset("label")))
    weightedDataset
  }

  def cleanColumns(df: DataFrame, columns: Array[String]): DataFrame = {
    import org.apache.spark.sql.functions._
    columns.foldLeft(df)( (dataframe, column) => {
      if (!dataframe.columns.contains(column)) dataframe.withColumn(column, lit(null))
      else dataframe
    }
    )
  }

  def getPipelineETL(newDf: DataFrame) = {
    if (File("model/pipelineETL").exists()){
      PipelineModel.load("model/pipelineETL")
    } else {

      val indexerAppOrSite =  new StringIndexer()
        .setStringOrderType("alphabetAsc")
        .setHandleInvalid("keep")
        .setInputCol("appOrSite")
        .setOutputCol("appOrSiteIndex")

      val indexerLabel = new StringIndexer()
        .setStringOrderType("alphabetAsc")
        .setHandleInvalid("keep")
        .setInputCol("label")
        .setOutputCol("labelIndex")

      val indexerExchange = new StringIndexer()
        .setStringOrderType("alphabetAsc")
        .setHandleInvalid("keep")
        .setInputCol("exchange")
        .setOutputCol("exchangeIndex")

      val indexerPublisher = new StringIndexer()
        .setStringOrderType("alphabetAsc")
        .setHandleInvalid("keep")
        .setInputCol("publisher")
        .setOutputCol("publisherIndex")

      val indexerNetwork = new StringIndexer()
        .setStringOrderType("alphabetAsc")
        .setHandleInvalid("keep")
        .setInputCol("network")
        .setOutputCol("networkIndex")

      val indexerCity = new StringIndexer()
        .setStringOrderType("alphabetAsc")
        .setHandleInvalid("keep")
        .setInputCol("city")
        .setOutputCol("cityIndex")

      val indexerMedia = new StringIndexer()
        .setStringOrderType("alphabetAsc")
        .setHandleInvalid("keep")
        .setInputCol("media")
        .setOutputCol("mediaIndex")

      val indexerOs = new StringIndexer()
        .setStringOrderType("alphabetAsc")
        .setHandleInvalid("keep")
        .setInputCol("os")
        .setOutputCol("osIndex")

      val indexerSize0 = new StringIndexer()
        .setStringOrderType("alphabetAsc")
        .setHandleInvalid("keep")
        .setInputCol("size0")
        .setOutputCol("size0Index")

      val indexerSize1 = new StringIndexer()
        .setStringOrderType("alphabetAsc")
        .setHandleInvalid("keep")
        .setInputCol("size1")
        .setOutputCol("size1Index")

      val indexerType = new StringIndexer()
        .setStringOrderType("alphabetAsc")
        .setHandleInvalid("keep")
        .setInputCol("type")
        .setOutputCol("typeIndex")

      val allCols = allIAB.toArray ++ Array("appOrSiteIndex", "exchangeIndex", "publisherIndex", "networkIndex", "cityIndex", "mediaIndex", "osIndex", "size0Index", "size1Index", "typeIndex", "bidfloor")

      val assembler = new VectorAssembler()
        .setInputCols(allCols)
        .setOutputCol("features")

      val pipelineEtl = new Pipeline()
        .setStages(Array(indexerAppOrSite, indexerLabel, indexerExchange, indexerPublisher,indexerNetwork, indexerCity, indexerMedia, indexerOs, indexerSize0, indexerSize1, indexerType, assembler))

      val model = pipelineEtl.fit(newDf)
      model.save("model/pipelineETL")
      model
    }
  }


  def indexString(df: DataFrame, c: String): DataFrame = {
    var newDf = df
    val indexer = new StringIndexer()
      .setStringOrderType("alphabetAsc")
      .setHandleInvalid("keep")
      .setInputCol(c)
      .setOutputCol(c+"I")

    newDf =
      indexer.fit(df)
        .transform(df)
        .drop(newDf.col(c))
        .withColumnRenamed(c+"I", c)
    newDf
  }

  def IndexStringArray(df: DataFrame, cols: Array[String]): DataFrame = {
    var newDf = df
    for (c <- cols){
      newDf = indexString(newDf, c)
    }
    newDf
  }

  def vectorize(df: DataFrame): DataFrame = {
    val assembler = new VectorAssembler()
      .setInputCols(df.columns.diff(Array("label")))
      .setOutputCol("features")

    assembler.transform(df).select("features", "label")
  }


  def removeColumns(df: DataFrame, cols: Array[String]): DataFrame = {
    var newDf = df
    for(c <- cols){
      newDf = newDf.drop(newDf.col(c))
    }
    newDf
  }

  def splitSize(df: DataFrame): DataFrame = {
    var newDf = df
    newDf = newDf.withColumn("size0", when(newDf.col("size").isNull, 0).otherwise(newDf.col("size")(0)))
    newDf = newDf.withColumn("size1", when(newDf.col("size").isNull, 0).otherwise(newDf.col("size")(1)))
    newDf = setNullableStateOfColumn(newDf, "size0", false)
    newDf = setNullableStateOfColumn(newDf, "size1", false)
    newDf = newDf.drop(newDf.col("size"))
    newDf
  }

  def splitAppOrSite(df: DataFrame): DataFrame = {
    var newDf = df
    newDf = newDf.withColumn("is_App", when(newDf.col("appOrSite")=== "app",1).otherwise(0))
    newDf = newDf.withColumn("is_Site", when(newDf.col("appOrSite")=== "site",1).otherwise(0))
    newDf = newDf.drop(newDf.col("appOrSite"))
    newDf
  }
  
  def cleanOS(df: DataFrame): DataFrame = {
    var newDf = df
    newDf = newDf
      .withColumn("OS", upper(newDf.col("os")))
      .drop(newDf.col("os"))
      .withColumnRenamed("OS", "os")
    newDf
  }

  def cleanType(df: DataFrame): DataFrame = {
    var newDf = df
    newDf = newDf.withColumn("typeC",
      when(newDf.col("type").isNull,4)
        .when(newDf.col("type")=== "CLICK",4)
        .otherwise(newDf.col("type")))
      .drop(newDf.col("type"))
      .withColumnRenamed("typeC", "type")

    newDf = setNullableStateOfColumn(newDf, "type", false)
    newDf
  }

  def cleanBidFloor(df: DataFrame): DataFrame = {
    var newDf = df
    newDf = newDf.withColumn("bidfloorC",
      when(newDf.col("bidfloor").isNull,0.0)
        .otherwise(newDf.col("bidfloor")))
      .drop(newDf.col("bidfloor"))
      .withColumnRenamed("bidfloorC", "bidfloor")

    newDf = setNullableStateOfColumn(newDf, "bidfloor", false)
    newDf
  }

  def labelToInt(df: DataFrame): DataFrame = {
    var newDf = df
    newDf = newDf.withColumnRenamed("label", "labelBool")
    newDf = newDf.withColumn("label", when(newDf.col("labelBool").isNull,0).when(newDf.col("labelBool")=== true,1).otherwise(0))
    newDf = newDf.drop(newDf.col("labelBool"))
    newDf
  }

  def labelToFloat(df: DataFrame): DataFrame = {
    var newDf = df
    newDf = newDf.withColumnRenamed("label", "labelInt")
    newDf = newDf.withColumn("label", when(newDf.col("labelInt")=== 1,1.0).otherwise(0.0))
    newDf = newDf.drop(newDf.col("labelInt"))
    newDf
  }

  def setNullableStateOfColumn( df: DataFrame, cn: String, nullable: Boolean) : DataFrame = {
    // get schema
    val schema = df.schema
    // modify [[StructField] with name `cn`
    val newSchema = StructType(schema.map {
      case StructField( c, t, _, m) if c.equals(cn) => StructField( c, t, nullable = nullable, m)
      case y: StructField => y
    })
    // apply new schema
    df.sqlContext.createDataFrame( df.rdd, newSchema )
  }

  def splitInterests(df: DataFrame): DataFrame = {
    var newDf = df

    for(iab <- allIAB){
      newDf = newDf.withColumn(iab, when(newDf.col("interests").contains(iab),1.0).otherwise(0.0))
    }
    newDf = newDf.drop("interests")
    newDf
  }

  val allIAB = List(
    "IAB1","IAB1-1","IAB1-2","IAB1-3","IAB1-4","IAB1-5","IAB1-6","IAB1-7",
    "IAB2", "IAB2-1", "IAB2-2","IAB2-3","IAB2-4","IAB2-5","IAB2-6","IAB2-7","IAB2-8","IAB2-9","IAB2-10","IAB2-11","IAB2-12","IAB2-13","IAB2-14","IAB2-15","IAB2-16","IAB2-17","IAB2-18","IAB2-19","IAB2-20","IAB2-21","IAB2-22","IAB2-23",
    "IAB3", "IAB3-1", "IAB3-2","IAB3-3","IAB3-4","IAB3-5","IAB3-6","IAB3-7","IAB3-8","IAB3-9","IAB3-10","IAB3-11","IAB3-12",
    "IAB4", "IAB4-1", "IAB4-2","IAB4-3","IAB4-4","IAB4-5","IAB4-6","IAB4-7","IAB2-8","IAB4-9","IAB4-10","IAB4-11",
    "IAB5", "IAB5-1", "IAB5-2","IAB5-3","IAB5-4","IAB5-5","IAB5-6","IAB5-7","IAB5-8","IAB5-9","IAB5-10","IAB5-11","IAB5-12","IAB5-13","IAB5-14","IAB5-15",
    "IAB6", "IAB6-1", "IAB6-2","IAB6-3","IAB6-4","IAB6-5","IAB6-6","IAB6-7","IAB6-8","IAB6-9",
    "IAB9", "IAB7-1", "IAB7-2","IAB7-3","IAB7-4","IAB7-5","IAB7-6","IAB7-7","IAB7-8","IAB7-9","IAB7-10","IAB7-11","IAB7-12","IAB7-13","IAB7-14","IAB7-15","IAB7-16","IAB7-17","IAB7-18","IAB7-19","IAB7-20","IAB7-21","IAB7-22","IAB7-23", "IAB7-24", "IAB7-25","IAB7-26","IAB7-27","IAB7-28","IAB7-29","IAB7-30","IAB7-31","IAB7-32","IAB7-33","IAB7-34","IAB7-35","IAB7-36","IAB7-37","IAB7-38","IAB7-39","IAB7-40","IAB7-41","IAB7-42","IAB7-43","IAB7-44","IAB7-45",
    "IAB8", "IAB8-1", "IAB8-2","IAB8-3","IAB8-4","IAB8-5","IAB8-6","IAB8-7","IAB8-8","IAB8-9","IAB8-10","IAB8-11","IAB8-12","IAB8-13","IAB8-14","IAB8-15","IAB8-16","IAB8-17","IAB8-18",
    "IAB9", "IAB9-1", "IAB9-2","IAB9-3","IAB9-4","IAB9-5","IAB9-6","IAB9-7","IAB9-8","IAB9-9","IAB9-10","IAB9-11","IAB9-12","IAB9-13","IAB9-14","IAB9-15","IAB9-16","IAB9-17","IAB9-18","IAB9-19","IAB9-20","IAB9-21","IAB9-22","IAB9-23", "IAB9-24", "IAB9-25","IAB9-26","IAB9-27","IAB9-28","IAB9-29","IAB9-30","IAB9-31",
    "IAB10", "IAB10-1", "IAB10-2","IAB10-3","IAB10-4","IAB10-5","IAB10-6","IAB10-7","IAB10-8","IAB10-9",
    "IAB11", "IAB11-1", "IAB11-2","IAB11-3","IAB11-4","IAB11-5",
    "IAB12", "IAB12-1", "IAB12-2","IAB12-3",
    "IAB13", "IAB13-1", "IAB13-2","IAB13-3","IAB13-4","IAB13-5","IAB13-6","IAB13-7","IAB13-8","IAB13-9","IAB13-10","IAB13-11","IAB13-12",
    "IAB14", "IAB14-1", "IAB14-2","IAB14-3","IAB14-4","IAB14-5","IAB14-6","IAB14-7","IAB14-8",
    "IAB15", "IAB15-1", "IAB15-2","IAB15-3","IAB15-4","IAB15-5","IAB15-6","IAB15-7","IAB15-8","IAB15-9","IAB15-10",
    "IAB16", "IAB16-1", "IAB16-2","IAB16-3","IAB16-4","IAB16-5","IAB16-6","IAB16-7",
    "IAB17", "IAB17-1", "IAB17-2","IAB17-3","IAB17-4","IAB17-5","IAB17-6","IAB17-7","IAB17-8","IAB17-9","IAB17-10","IAB17-11","IAB17-12","IAB17-13","IAB17-14","IAB17-15","IAB17-16","IAB17-17","IAB17-18","IAB17-19","IAB17-20","IAB17-21","IAB17-22","IAB17-23", "IAB17-24", "IAB17-25","IAB17-26","IAB17-27","IAB17-28","IAB17-29","IAB17-30","IAB17-31","IAB17-32","IAB17-33","IAB17-34","IAB17-35","IAB17-36","IAB17-37","IAB17-38","IAB17-39","IAB17-40","IAB17-41","IAB17-42","IAB17-43","IAB17-44",
    "IAB18", "IAB18-1", "IAB18-2","IAB18-3","IAB18-4","IAB18-5","IAB18-6",
    "IAB19", "IAB19-1", "IAB19-2","IAB19-3","IAB19-4","IAB19-5","IAB19-6","IAB19-7","IAB19-8","IAB19-9","IAB19-10","IAB19-11","IAB19-12","IAB19-13","IAB19-14","IAB19-15","IAB19-16","IAB19-17","IAB19-18","IAB19-19","IAB19-20","IAB19-21","IAB19-22","IAB19-23", "IAB19-24", "IAB19-25","IAB19-26","IAB19-27","IAB19-28","IAB19-29","IAB19-30","IAB19-31","IAB19-32","IAB19-33","IAB19-34","IAB19-35","IAB19-36",
    "IAB20", "IAB20-1", "IAB20-2","IAB20-3","IAB20-4","IAB20-5","IAB20-6","IAB20-7","IAB20-8","IAB20-9","IAB20-10","IAB20-11","IAB20-12","IAB20-13","IAB20-14","IAB20-15","IAB20-16","IAB20-17","IAB20-18","IAB20-19","IAB20-20","IAB20-21","IAB20-22","IAB20-23", "IAB20-24", "IAB20-25","IAB20-26","IAB20-27",
    "IAB21", "IAB21-1", "IAB21-2","IAB21-3",
    "IAB22", "IAB22-1", "IAB22-2","IAB22-3","IAB22-4",
    "IAB23", "IAB23-1", "IAB23-2","IAB23-3","IAB23-4","IAB23-5","IAB23-6","IAB23-7","IAB23-8","IAB23-9","IAB23-10",
    "IAB25", "IAB25-1", "IAB25-2","IAB25-3","IAB25-4","IAB25-5","IAB25-6","IAB25-7",
    "IAB26", "IAB26-1", "IAB26-2","IAB26-3","IAB26-4"
  )

}
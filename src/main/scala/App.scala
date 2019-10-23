import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, StringIndexerModel, VectorIndexer}
import org.apache.spark.sql._

object App {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("WI").setMaster("local")

    val spark = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()

    var dataframe: DataFrame = getDataframe(spark)
    // Preprocessing
    dataframe = dataframe.withColumn("size0", dataframe.col("size")(0))
    dataframe = dataframe.withColumn("size1", dataframe.col("size")(1))
    dataframe = dataframe.drop(dataframe.col("size"))
    dataframe = dataframe.drop(dataframe.col("impid"))
    dataframe = dataframe.drop(dataframe.col("exchange"))


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

    // IAB SHIT (don't use when not the code)
    for(iab <- allIAB){
      dataframe = dataframe.withColumn(iab,dataframe.col("interests").contains(iab))
    }
    dataframe = dataframe.drop(dataframe.col("interests"))

    val someData = dataframe.randomSplit(Array(10,90))
    val testData = someData(0)
    val trainData = someData(1)

    /*
        val newTest =
          test
              .select("appOrSite","bidfloor", "city", "interests", "media", "network", "publisher", "size", "timestamp", "type", "os", "label")
              .printSchema()

        val getAllCities = test
            .select("city")
            .distinct()
            .collect
            .foreach( println(_))
    */

    testData.printSchema()
   // val df_numerics = indexCol(trainData, trainData.columns)
   // val dfHot = encodeCol(df_numerics,Array("appOrSite","bidfloor", "city", "interests", "media", "network", "publisher", "size", "timestamp", "type", "os"))

    // dfHot.show(15)

    /**val labelIndexer = new StringIndexer()
        .setInputCol("label")
        .setOutputCol("indexedLabel")
        .fit(dataframe)

    val featureIndexer = new VectorIndexer()
        .setInputCol("features")
        .setOutputCol("indexedFeatures")
        .fit(dataframe)
    **/
    spark.stop()
  }

  def getDataframe(spark: SparkSession): DataFrame =  {
    val path = "data.json"
    spark.read
      .option("inferSchema", value = true)
      .json(path)
  }

  def encodeCol(df: DataFrame, cols: Array[String]): DataFrame = {
    var newDf = df
      val encoder = new OneHotEncoderEstimator()
        .setInputCols(cols)
        .setOutputCols(cols.map( s => s +"-encoded"))
        .setDropLast(false)
    newDf
  }

  def indexCol(df: DataFrame, cols: Array[String]): DataFrame = {
    var newDf = df
    for(c <- cols){
      val si = new StringIndexer()
        .setInputCol(c)
        .setOutputCol(c + "-num")
      val sm: StringIndexerModel = si.fit(newDf)
      newDf = sm.transform(newDf).drop(c)
      newDf = newDf.withColumnRenamed(c + "-num", c)
    }
    newDf
  }

}

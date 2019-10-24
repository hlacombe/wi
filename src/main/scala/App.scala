import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, StringIndexerModel, VectorIndexer}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.{StructField, StructType}

object App {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("WI").setMaster("local")

    val spark = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()

    val sc = spark.sparkContext
    var dataframe: DataFrame = getDataframe(spark)

    // Preprocessing

    dataframe = Etl.splitSize(dataframe)
    dataframe = Etl.splitAppOrSite(dataframe)
    dataframe = Etl.cleanType(dataframe)
    dataframe = Etl.cleanBidFloor(dataframe)
    dataframe = Etl.removeColumns(dataframe, Array("network", "user", "timestamp", "exchange", "impid"))
    dataframe = Etl.replaceNullStringColumns(dataframe, Array("city", "publisher", "os", "media"))
    // dataframe = Etl.splitInterests(dataframe)
    dataframe = Etl.labelToInt(dataframe)
    dataframe.printSchema()
    dataframe.show(1)
    /*

    val assembler = new VectorAssembler()
      .setInputCols(dataframe.columns.diff(Array("label")))
      .setOutputCol("features")

    val lpoints = assembler.transform(dataframe).select("features", "label")

    val splits = lpoints.randomSplit(Array(10,90))
    val testData = splits(0).cache()
    val trainData = splits(1).cache()

    val layers = Array[Int](dataframe.columns.size-1, 10, 10, 2)

    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    // train the model
    val model = trainer.fit(trainData)

    // compute accuracy on the test set
    val result = model.transform(testData)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
*/
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

    // val df_numerics = indexCol(testData, Array("appOrSite", "city", "os", "publisher"))
    // df_numerics.show(1)
    // val dfHot = encodeCol(df_numerics, testData.columns)

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
      val indexer = new StringIndexer()
        .setInputCol(c)
        .setOutputCol(c+"_index")

      newDf = indexer.fit(newDf).transform(newDf).drop(newDf.col(c)).withColumnRenamed(c + "_index", c)
    }
    newDf
  }

}

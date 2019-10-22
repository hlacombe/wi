package org.apache.spark.examples.sql
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

    val dataframe: DataFrame = getDataframe(spark)
    val someData =
      dataframe.randomSplit(Array(10,90))

    val test = someData(0)
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

    val df_numerics = indexCol(test,Array("appOrSite","bidfloor", "city", "interests", "media", "network", "publisher", "size", "timestamp", "type", "os"))
    val dfHot = encodeCol(df_numerics,Array("appOrSite","bidfloor", "city", "interests", "media", "network", "publisher", "size", "timestamp", "type", "os"))

    dfHot.show(15)

    val labelIndexer = new StringIndexer()
        .setInputCol("label")
        .setOutputCol("indexedLabel")
        .fit(dataframe)

    val featureIndexer = new VectorIndexer()
        .setInputCol("features")
        .setOutputCol("indexedFeatures")
        .fit(dataframe)

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

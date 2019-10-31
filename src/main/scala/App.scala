import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql._

object App {

  def main(args: Array[String]): Unit = {

    Logger
      .getLogger("org")
      .setLevel(Level.ERROR)

    Logger
      .getLogger("akka")
      .setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setAppName("WI")
      .setMaster("local")

    val spark = SparkSession
      .builder()
      .config(conf)
      .getOrCreate()

    var dataframe = spark.read
      .option("inferSchema", value = true)
      .json("small-data.json")

    // ETL
    dataframe = Etl.splitSize(dataframe)
    dataframe = Etl.splitAppOrSite(dataframe)
    dataframe = Etl.cleanType(dataframe)
    dataframe = Etl.cleanBidFloor(dataframe)
    dataframe = Etl.removeColumns(dataframe, Array("network", "user", "timestamp", "exchange", "impid"))
    dataframe = Etl.replaceNullStringColumns(dataframe, Array("city", "publisher", "os", "media"))
    dataframe = Etl.splitInterests(dataframe)
    //dataframe = Etl.removeColumns(dataframe, Array("interests"))
    dataframe = Etl.labelToInt(dataframe)
    dataframe = Etl.IndexStringArray(dataframe, Array("city", "publisher", "os", "media", "size0", "size1", "type"))
    val dataframeV = Etl.vectorize(dataframe)


    // Perceptron //
    val splits = dataframeV.randomSplit(Array(20,80))
    val testData = splits(0).cache()
    val trainData = splits(1).cache()

    MultilayerPerceptron.train(trainData, dataframe, "model/Perceptron")
    MultilayerPerceptron.predict(testData, "model/Perceptron")

    // Random Forest //
    RandomForest.predict(dataframeV, "model/RandomForest")

    // Other Methode

    spark.stop()
  }

}

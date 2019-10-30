import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
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

    val sc = spark.sparkContext
    var dataframe: DataFrame = getDataframe(spark)

    // Preprocessing

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

    RandomForest.predict(dataframeV)

    spark.stop()
  }

  def getDataframe(spark: SparkSession): DataFrame =  {
    val path = "small-data.json"
    spark.read
      .option("inferSchema", value = true)
      .json(path)
  }
}

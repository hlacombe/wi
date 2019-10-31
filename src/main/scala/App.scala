import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql._
import better.files._

object App {

  def main(args: Array[String]): Unit = {

    Logger
      .getLogger("org")
      .setLevel(Level.ERROR)

    Logger
      .getLogger("akka")
      .setLevel(Level.ERROR)

    if (args.length != 2){
      println(s"Invalid argument number [train|predict] <json file>")
    } else {
      val action = args(0)
      val fileJson = args(1)
      if (!File(fileJson).exists()){
        println(s"Invalid json file '${fileJson}'")
      } else {
        if (action != "train" && action != "predict"){
          println(s"Invalid argument [train|predict] <json file>")
        } else {
          val conf = new SparkConf()
            .setAppName("WI")
            .setMaster("local")

          val spark = SparkSession
            .builder()
            .config(conf)
            .getOrCreate()

          var dataframe = spark.read
            .option("inferSchema", value = true)
            .json(fileJson)

          val (dataframeV, nbFeatures) = Etl.etlVectorize(dataframe)
          val modelFolder = getModelFolder()

          action match {
            case "train" =>
              println(s"Training on '${fileJson}'")

              if (modelFolder.exists()){
                modelFolder.delete()
              }

              MultilayerPerceptron.train(dataframeV, nbFeatures, "model/Perceptron")
              RandomForest.train(dataframeV, "model/RandomForest")


            case "predict" =>
              println(s"Prediction on '${fileJson}'")
              if(!modelFolder.exists()){
                "No model pre-trained"
              } else {
                MultilayerPerceptron.predict(dataframeV, "model/Perceptron")
                RandomForest.predict(dataframeV, "model/RandomForest")


              }


            case _ => println("Error")
          }

          spark.stop()
        }

      }

      /*
      // Perceptron //
      val splits = dataframeV.randomSplit(Array(20,80))
      val testData = splits(0).cache()
      val trainData = splits(1).cache()

      MultilayerPerceptron.train(trainData, dataframe, "model/Perceptron")
      MultilayerPerceptron.predict(testData, "model/Perceptron")

      // Random Forest //
      RandomForest.predict(dataframeV, "model/RandomForest")

      // Other Methode



       */
    }

  }

  def getModelFolder(): File = {
    return File("model")
  }

}

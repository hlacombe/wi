import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql._
import better.files._
import org.apache.spark.sql.functions._

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

          val (dataframeV, nbFeatures) = Etl.etlVectorize(dataframe, action)
          val modelFolder = getModelFolder()

          action match {
            case "train" =>
              println(s"Training on '${fileJson}'")

              if (modelFolder.exists()){
                modelFolder.delete()
              }

              //MultilayerPerceptron.train(dataframeV, nbFeatures, "model/Perceptron")
              //RandomForest.train(dataframeV, "model/RandomForest")
              //LogisticReg.train(dataframeV, "model/logisticRegression")
              val splits = dataframeV.randomSplit(Array(0.8, 0.2), seed = 123L)
              val trainingData = splits(0).cache()
              val testData = splits(1)
              LogisticReg.train(trainingData, "model/logisticRegression")
              LogisticReg.predict(testData, "model/logisticRegression")

            case "predict" =>
              println(s"Prediction on '${fileJson}'")
              if(!modelFolder.exists()){
                "No model pre-trained"
              } else {
                //MultilayerPerceptron.predict(dataframeV, "model/Perceptron")
                //RandomForest.predict(dataframeV, "model/RandomForest")
                val predictions = LogisticReg.predict(dataframeV, "model/logisticRegression")
                val stringify = udf((vs: Seq[String]) => vs match {
                  case null => null
                  case _    => s"""[${vs.mkString(",")}]"""
                })

                val df1 = predictions.withColumn("id", monotonically_increasing_id)
                val df2 = dataframe.withColumn("id", monotonically_increasing_id)
                val df3 = df1.join(df2, df1("id") === df2("id"), "outer")
                  .drop("id")
                  .withColumn("size", stringify(col("size")))
                  .repartition(1)
                  .write
                  .format("com.databricks.spark.csv")
                  .option("header", "true")
                  .save("output")
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

import org.apache.spark.ml.classification.{ MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.DataFrame
import java.util.Calendar

object MultilayerPerceptron {

  def train(dataframeV: DataFrame, dataframe: DataFrame, modelPath: String): Unit ={
    val startTime = Calendar.getInstance.getTime
    val layers = Array[Int](dataframe.columns.size-1, 10, 10, 2)

    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    val model = trainer.fit(dataframeV)
    model.save(modelPath)
    val endTime = Calendar.getInstance.getTime

    println("Start : " + startTime.toString)
    println("End : " + endTime.toString)
  }

  def predict(dataToPredict: DataFrame, modelPath: String): Unit = {
    val modelLoaded = MultilayerPerceptronClassificationModel.load(modelPath)
    val result = modelLoaded.transform(dataToPredict)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
  }

}

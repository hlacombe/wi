import org.apache.spark.ml.classification.{ MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.DataFrame

object MultilayerPerceptron {

  def train(dataframeV: DataFrame, nbFeatures: Int, modelPath: String): Unit ={
    val layers = Array[Int](nbFeatures, 10, 10, 2)

    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    val model = trainer.fit(dataframeV)
    model.save(modelPath)
    println("Perceptron model saved")
  }

  def predict(dataToPredict: DataFrame, modelPath: String): Unit = {
    val modelLoaded = MultilayerPerceptronClassificationModel.load(modelPath)
    val result = modelLoaded.transform(dataToPredict)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println(s"Accuracy Perceptron = ${evaluator.evaluate(predictionAndLabels)}")
  }

}

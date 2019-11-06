import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object LogisticReg {
  def train(dataframeV: DataFrame, modelPath: String): Unit = {
    val logRegModel = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxIter(10)
      .fit(dataframeV)

    println(s"Intercept: ${logRegModel.intercept}")


    logRegModel.save(modelPath)
    println("Logistic Regression model saved")
  }

  def predict(dataToPredict: DataFrame, modelPath: String): Unit = {
    val df = dataToPredict

    val modelLoaded = LogisticRegressionModel.load(modelPath)
    val predictions = modelLoaded.transform(df)
    predictions.show(200)
    val predictionAndLabels = predictions.select("prediction", "labelIndex").rdd.map(x => (x.get(0).asInstanceOf[Double], x.get(1).asInstanceOf[Double]))

    val metrics = new MulticlassMetrics(predictionAndLabels)
    println(s"Weighted precision= ${metrics.weightedPrecision}")
    println(s"Weighted recall= ${metrics.weightedRecall}")

    println("Confusion matrix:")
    println(metrics.confusionMatrix)
  }
}
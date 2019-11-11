import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.DataFrame

object LogisticReg {
  def train(dataframeV: DataFrame, modelPath: String): Unit = {

    val logRegModel = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxIter(250)
      .setWeightCol("classWeightCol")
      .setFamily("binomial")
      .fit(dataframeV)

    println(s"Intercept: ${logRegModel.intercept}")

    logRegModel.save(modelPath)
    println("Logistic Regression model saved")
  }

  def predict(dataToPredict: DataFrame, modelPath: String): DataFrame = {
    val df = dataToPredict

    val modelLoaded = LogisticRegressionModel.load(modelPath)
    val predictions = modelLoaded.transform(df)

    /*
    val predictionAndLabels = predictions.select("prediction", "labelIndex").rdd.map(x => (x.get(0).asInstanceOf[Double], x.get(1).asInstanceOf[Double]))
    val metrics = new MulticlassMetrics(predictionAndLabels)
    println(s"Weighted precision= ${metrics.weightedPrecision}")
    println(s"Weighted recall= ${metrics.weightedRecall}")

    println("Confusion matrix:")
    println(metrics.confusionMatrix)*/

    predictions.select("prediction")
  }
}
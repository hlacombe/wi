import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.DataFrame

object Bayes {

  def predict(dataframeV: DataFrame, modelPath: String): Unit ={

    val splits = dataframeV.randomSplit(Array(70,30),seed = 1234)

    val testData = splits(0).cache()
    val trainData = splits(1).cache()

    val model = new NaiveBayes()
      .setFeaturesCol("features")
      .setLabelCol("label")
      //.setWeightCol("classWeightCol")
      .setThresholds(Array(1, 100))
      .fit(trainData)

    val predictions = model.transform(testData)
    predictions.show()

    val predictionAndLabels = predictions.select("prediction", "labelIndex").rdd.map(x => (x.get(0).asInstanceOf[Double], x.get(1).asInstanceOf[Double]))

    val metrics = new MulticlassMetrics(predictionAndLabels)
    println(s"Weighted precision= ${metrics.weightedPrecision}")
    println(s"Weighted recall= ${metrics.weightedRecall}")

    println("Confusion matrix:")
    println(metrics.confusionMatrix)
  }


}

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.DataFrame

object Bayes {

  def predict(dataframeV: DataFrame, modelPath: String): Unit ={

    val splits = dataframeV.randomSplit(Array(70,30))

    val testData = splits(0).cache()
    val trainData = splits(1).cache()

    // Train a NaiveBayes model.
    val model = new NaiveBayes()
      .fit(trainData)

    // Select example rows to display.
    val predictions = model.transform(testData)
    predictions.show()

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test set accuracy = " + accuracy)
  }


}

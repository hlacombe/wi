import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.DataFrame

object MultilayerPerceptron {

  def predict(dataframeV: DataFrame, dataframe: DataFrame): Unit ={
    val splits = dataframeV.randomSplit(Array(10,90))
    val testData = splits(0).cache()
    val trainData = splits(1).cache()

    //MultilayerPerceptronClassificationModel model = MultilayerPerceptronClassificationModel.load(getSavedModelPath());

    val layers = Array[Int](dataframe.columns.size-1, 10, 10, 2)

    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    val model = trainer.fit(trainData)
    model.save("modelSpark")
    // compute accuracy on the test set
    val result = model.transform(testData)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
  }

}

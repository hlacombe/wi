import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.DataFrame

object RandomForest {

  def train(dataframeV: DataFrame, modelPath: String): Unit = {
    val labelIndexer = new StringIndexer()
      .setStringOrderType("alphabetAsc")
      .setHandleInvalid("keep")
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(dataframeV)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
      .fit(dataframeV)

    val dt = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxBins(100)
      .setNumTrees(10000)
      .setMaxDepth(30)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    val model = pipeline.fit(dataframeV)
    model.save(modelPath)
    println("RandomForest model Saved")
  }

  def predict(dataframeV: DataFrame, modelPath: String): Unit = {
    val model = PipelineModel.load(modelPath)
    val predictions = model.transform(dataframeV)
    
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy RandomForest = " + (accuracy))

    //val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    //println("Learned classification tree model:\n" + treeModel.toDebugString)

    //val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    //println("Learned classification forest model:\n" + rfModel.toDebugString)
  }
}

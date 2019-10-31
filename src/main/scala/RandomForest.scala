import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.DataFrame

object RandomForest {
  def predict(dataframeV: DataFrame, modelPath: String): Unit ={

    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(dataframeV)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
      .fit(dataframeV)

    val splits = dataframeV.randomSplit(Array(70,30))

    val testData = splits(0).cache()
    val trainData = splits(1).cache()

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

    val model = pipeline.fit(trainData)
    model.save(modelPath)

    val modelLoaded = PipelineModel.load(modelPath)
    val predictions = model.transform(testData)
    

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    //val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    //println("Learned classification tree model:\n" + treeModel.toDebugString)

    //val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    //println("Learned classification forest model:\n" + rfModel.toDebugString)
  }
}

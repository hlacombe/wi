import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.DataFrame

object LogisticRegressionModel {

  def predict(dataframeV: DataFrame, modelPath: String): Unit ={

    val splits = dataframeV.randomSplit(Array(30,70))

    val testData = splits(0).cache()
    val trainData = splits(1).cache()

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(trainData)

    // Print the coefficients and intercept for logistic regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
  }
}

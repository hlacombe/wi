name := "wi_spark-project"

version := "0.1"

scalaVersion := "2.12.10"

libraryDependencies := Seq(
  "org.apache.spark" %% "spark-core" % "2.4.4",
  "org.apache.spark" %% "spark-sql" % "2.4.4",
  "org.apache.spark" %% "spark-mllib" % "2.4.4"
)

// https://mvnrepository.com/artifact/org.apache.spark/spark-core
// https://mvnrepository.com/artifact/org.apache.spark/spark-sql
// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib

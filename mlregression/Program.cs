using System;
using System.IO;

using mldeepdivelib.Common;

using mlregression.Structures;

using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Normalizers;

namespace mlregression
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);

            switch (args[0])
            {
                case "build":
                    TrainModel<EmploymentHistory>(mlContext, args[1], args[2]);
                    break;
                case "predict":
                    Predict(mlContext, args[1], args[2]);
                    break;
            }
        }

        private static void Predict(MLContext mlContext, string modelPath, string predictionFilePath)
        {
            var employmentRecord = Newtonsoft.Json.JsonConvert.DeserializeObject<EmploymentHistory>(File.ReadAllText(predictionFilePath));

            ITransformer trainedModel;

            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                trainedModel = mlContext.Model.Load(stream);
            }
            
            var predFunction = trainedModel.MakePredictionFunction<EmploymentHistory, EmploymentHistoryPrediction>(mlContext);
            
            var resultprediction = predFunction.Predict(employmentRecord);
            
            Console.WriteLine($"Predicted Duration (in months): {resultprediction.DurationInMonths:0.#}");
        }

        private static void TrainModel<T>(MLContext mlContext, string trainDataPath, string modelPath) {
            var textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = false,
                Column = Activator.CreateInstance<T>().ToColumns()
            });
            
            var baseTrainingDataView = textLoader.Read(trainDataPath);
            
            var trainingDataView = mlContext.Data.FilterByColumn(baseTrainingDataView, "DurationInMonths", 1, 150);
            
            var dataProcessPipeline = mlContext.Transforms.CopyColumns("DurationInMonths", "Label")
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding("PositionName", "PositionNameEncoded"))
                            .Append(mlContext.Transforms.Normalize("IsMarried", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                            .Append(mlContext.Transforms.Normalize("BSDegree", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                            .Append(mlContext.Transforms.Normalize(inputName: "MSDegree", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                            .Append(mlContext.Transforms.Normalize(inputName: "YearsExperience", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                            .Append(mlContext.Transforms.Normalize(inputName: "AgeAtHire", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                            .Append(mlContext.Transforms.Concatenate("Features", "PositionNameEncoded", "IsMarried", "BSDegree", "MSDegree", "YearsExperience", "AgeAtHire"));
                                     
            var trainer = mlContext.Regression.Trainers.StochasticDualCoordinateAscent();
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            
            var trainedModel = trainingPipeline.Fit(trainingDataView);
            
            using (var fs = File.Create(modelPath))
            {
                trainedModel.SaveTo(mlContext, fs);
            }

            Console.WriteLine($"Saved model to {modelPath}");
        }
    }
}
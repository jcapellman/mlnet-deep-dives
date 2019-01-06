using System;
using System.IO;
using System.Linq;

using mlregression.Structures;

using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
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
                    TrainModel(mlContext, args[1], args[2]);
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

        private static void TrainModel(MLContext mlContext, string trainDataPath, string modelPath) {
            var textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = false,
                Column = new[]
                {
                    new TextLoader.Column("PositionName", DataKind.Text, 0),
                    new TextLoader.Column("DurationInMonths", DataKind.R4, 1),
                    new TextLoader.Column("IsMarried", DataKind.R4, 2),
                    new TextLoader.Column("BSDegree", DataKind.R4, 3),
                    new TextLoader.Column("MSDegree", DataKind.R4, 4),
                    new TextLoader.Column("YearsExperience", DataKind.R4, 5),
                    new TextLoader.Column("AgeAtHire", DataKind.R4, 6)
                }
            });

            var baseTrainingDataView = textLoader.Read(trainDataPath);
            
            var cnt = baseTrainingDataView.GetColumn<float>(mlContext, "DurationInMonths").Count();
            var trainingDataView = mlContext.Data.FilterByColumn(baseTrainingDataView, "DurationInMonths", lowerBound: 1, upperBound: 150);
            var cnt2 = trainingDataView.GetColumn<float>(mlContext, "DurationInMonths").Count();
            
            var dataProcessPipeline = mlContext.Transforms.CopyColumns("DurationInMonths", "Label")
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding("PositionName", "PositionNameEncoded"))
                            .Append(mlContext.Transforms.Normalize("IsMarried", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                            .Append(mlContext.Transforms.Normalize("BSDegree", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                            .Append(mlContext.Transforms.Normalize(inputName: "MSDegree", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                            .Append(mlContext.Transforms.Normalize(inputName: "YearsExperience", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                            .Append(mlContext.Transforms.Normalize(inputName: "AgeAtHire", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                            .Append(mlContext.Transforms.Concatenate("Features", "PositionNameEncoded", "IsMarried", "BSDegree", "MSDegree", "YearsExperience", "AgeAtHire"));
                                     
            var trainer = mlContext.Regression.Trainers.StochasticDualCoordinateAscent(labelColumn: "Label", featureColumn: "Features");
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
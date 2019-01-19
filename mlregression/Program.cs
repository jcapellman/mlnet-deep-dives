using System;
using System.IO;

using mldeepdivelib.Common;

using mlregression.Structures;

using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Transforms.Normalizers;

namespace mlregression
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);

            Console.Clear();

            switch (args[0])
            {
                case "build":
                    TrainModel<EmploymentHistory>(mlContext, args[1], args[2]);
                    break;
                case "predict":
                    var prediction = Predict<EmploymentHistory, EmploymentHistoryPrediction>(mlContext, args[1], args[2]);

                    Console.WriteLine($"Predicted Duration (in months): {prediction.DurationInMonths:0.#}");

                    break;
            }
        }

        private static TK Predict<T, TK>(MLContext mlContext, string modelPath, string predictionFilePath) where T : class where TK : class, new()
        {
            var data = File.ReadAllText(predictionFilePath);

            var predictionData = Newtonsoft.Json.JsonConvert.DeserializeObject<T>(data);

            Console.WriteLine($"Data:{Environment.NewLine}{data}{Environment.NewLine}");

            ITransformer trainedModel;

            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                trainedModel = mlContext.Model.Load(stream);
            }
            
            var predFunction = trainedModel.CreatePredictionEngine<T, TK>(mlContext);

            return predFunction.Predict(predictionData);
        }

        private static void TrainModel<T>(MLContext mlContext, string trainDataPath, string modelPath) {
            var modelObject = Activator.CreateInstance<T>();

            var textReader = mlContext.Data.CreateTextReader(columns: modelObject.ToColumns(), hasHeader: false, separatorChar: ',');

            var baseTrainingDataView = textReader.Read(trainDataPath);

            var label = modelObject.GetLabelAttributes();

            var trainingDataView = mlContext.Data.FilterByColumn(baseTrainingDataView, label.Name, label.Min, label.Max);

            var dataProcessPipeline = mlContext.Transforms.CopyColumns(label.Name, "Label")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("PositionName", "PositionNameEncoded"))
                .Append(mlContext.Transforms.Normalize("IsMarried", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(mlContext.Transforms.Normalize("BSDegree", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(mlContext.Transforms.Normalize(inputName: "MSDegree", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(mlContext.Transforms.Normalize(inputName: "YearsExperience", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(mlContext.Transforms.Normalize(inputName: "AgeAtHire", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(mlContext.Transforms.Normalize(inputName: "HasKids", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(mlContext.Transforms.Normalize(inputName: "WithinMonthOfVesting", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(mlContext.Transforms.Normalize(inputName: "DeskDecorations", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(mlContext.Transforms.Normalize(inputName: "LongCommute", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(mlContext.Transforms.Concatenate("Features", "PositionNameEncoded", "IsMarried", "BSDegree", "MSDegree", "YearsExperience", "AgeAtHire", "HasKids", "WithinMonthOfVesting", "DeskDecorations", "LongCommute"));
            
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
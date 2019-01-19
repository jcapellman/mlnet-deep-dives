using System;
using System.IO;

using Microsoft.ML;

using mldeepdivelib.Common;

using ThreatClassifier.Structures;

namespace ThreatClassifier
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);

            Console.Clear();

            switch (args[0])
            {
                case "trainmodel":
                    TrainModel<ThreatInformation>(mlContext, args[1], args[2]);
                    break;
            }
        }

        private static void TrainModel<T>(MLContext mlContext, string trainDataPath, string modelPath)
        {
            var modelObject = Activator.CreateInstance<T>();

            var textReader = mlContext.Data.CreateTextReader(columns: modelObject.ToColumns(), hasHeader: false, separatorChar: ',');

            var dataView = textReader.Read(trainDataPath);
            
            var pipeline = mlContext.Transforms
                .Concatenate(Constants.FEATURE_COLUMN_NAME, modelObject.ToColumnNames())
                .Append(mlContext.Clustering.Trainers.KMeans(Constants.FEATURE_COLUMN_NAME, clustersCount: 3));

            var trainedModel = pipeline.Fit(dataView);

            using (var fs = File.Create(modelPath))
            {
                trainedModel.SaveTo(mlContext, fs);
            }

            Console.WriteLine($"Saved model to {modelPath}");
        }
    }
}
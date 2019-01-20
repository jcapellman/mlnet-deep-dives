using System;
using System.IO;
using System.Linq;
using System.Text;

using Microsoft.ML;

using mldeepdivelib.Common;
using mldeepdivelib.Enums;
using mldeepdivelib.Helpers;

using ThreatClassifier.Common;
using ThreatClassifier.Structures;

namespace ThreatClassifier
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);

            Console.Clear();

            if (!Enum.TryParse(typeof(MLOperations), args[0], out var mlOperation))
            {
                Console.WriteLine($"{args[0]} is an invalid argument");

                Console.WriteLine("Available Options:");

                Console.WriteLine(string.Join(", ", Enum.GetNames(typeof(MLOperations))));

                return; 
            }

            switch (mlOperation)
            {
                case MLOperations.train:
                    TrainModel<ThreatInformation>(mlContext, args[1], args[2]);
                    break;
                case MLOperations.predict:
                    var extraction = FeatureExtractFile(args[2], true);

                    if (extraction == null)
                    {
                        return;
                    }

                    Console.WriteLine($"Predicting on {args[2]}:");

                    var prediction = Predictor.Predict<ThreatInformation, ThreatPredictor>(mlContext, args[1], extraction);

                    PrettyPrintResult(prediction);
                    break;
                case MLOperations.featureextraction:
                    FeatureExtraction(args[1], args[2]);
                    break;
            }
        }

        private static void PrettyPrintResult(ThreatPredictor prediction)
        {
            var threatType = (ThreatTypes)prediction.ThreatClusterId;

            Console.WriteLine($"Threat Type: {threatType}");
            
            Console.WriteLine("Category Breakdown:");

            for (var x = 0; x < prediction.Distances.Length; x++)
            {
                threatType = (ThreatTypes)x;

                Console.WriteLine($"{threatType} - {prediction.Distances[x]}%");
            }
        }

        private static ThreatInformation FeatureExtractFile(string filePath, bool forPrediction = false)
        {
            var peFile = new PeNet.PeFile(filePath);
            
            var information = new ThreatInformation
            {
                NumberImports = peFile.ImageResourceDirectory.NumberOfIdEntries,
                DataSizeInBytes = peFile.ImageSectionHeaders.FirstOrDefault()?.SizeOfRawData ?? 0.0f
            };
            
            foreach (var name in Enum.GetNames(typeof(ThreatTypes)))
            {
                if (!filePath.Contains(name))
                {
                    continue;
                }

                information.Classification = name;
            }

            if (!string.IsNullOrEmpty(information.Classification) || forPrediction)
            {
                return information;
            }

            Console.WriteLine($"{filePath} was not named properly");

            return null;
        }

        private static void FeatureExtraction(string rawDataFolder, string outputFile)
        {
            var startDate = DateTime.Now;

            var files = Directory.GetFiles(rawDataFolder);
            
            var sb = new StringBuilder();

            foreach (var filePath in files)
            {
                var extraction = FeatureExtractFile(filePath);

                if (extraction == null)
                {
                    continue;
                }

               sb.AppendLine(extraction.ToString());
            }

            File.WriteAllText(outputFile, sb.ToString());

            Console.WriteLine($"Feature Extraction completed in {DateTime.Now.Subtract(startDate).TotalMinutes} minutes to {outputFile}");
        }

        private static void TrainModel<T>(MLContext mlContext, string trainDataPath, string modelPath)
        {
            var modelObject = Activator.CreateInstance<T>();

            var textReader = mlContext.Data.CreateTextReader(columns: modelObject.ToColumns(), hasHeader: false, separatorChar: ',');

            var dataView = textReader.Read(trainDataPath);
            
            var pipeline = mlContext.Transforms
                .Concatenate(Constants.FEATURE_COLUMN_NAME, modelObject.ToColumnNames())
                .Append(mlContext.Clustering.Trainers.KMeans(
                Constants.FEATURE_COLUMN_NAME, 
                        clustersCount: Enum.GetNames(typeof(ThreatTypes)).Length));

            var trainedModel = pipeline.Fit(dataView);

            using (var fs = File.Create(modelPath))
            {
                trainedModel.SaveTo(mlContext, fs);
            }

            Console.WriteLine($"Saved model to {modelPath}");
        }
    }
}
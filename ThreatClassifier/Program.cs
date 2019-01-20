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

                return; 
            }

            switch (mlOperation)
            {
                case MLOperations.train:
                    TrainModel<ThreatInformation>(mlContext, args[1], args[2]);
                    break;
                case MLOperations.predict:
                    var prediction = Predictor.Predict<ThreatInformation, ThreatPredictor>(mlContext, args[1], args[2]);

                    PrettyPrintResult(prediction);
                    break;
                case MLOperations.featureextraction:
                    FeatureExtraction(args[1], args[2]);
                    break;
            }
        }

        private static void PrettyPrintResult(ThreatPredictor prediction)
        {
            var threatType = (ThreatTypes) (prediction.ThreatClusterId + 1);

            Console.WriteLine($"Threat Type: {threatType}");
            Console.WriteLine($"Cluster: {prediction.ThreatClusterId}");

            Console.WriteLine("Distances:");

            for (var x = 0; x <= prediction.Distances.Length; x++)
            {
                threatType = (ThreatTypes)(x + 1);

                Console.WriteLine($"{threatType} - {prediction.Distances[x]}%");
            }
        }

        private static void FeatureExtraction(string rawDataFolder, string outputFile)
        {
            var startDate = DateTime.Now;

            var files = Directory.GetFiles(rawDataFolder);

            var sb = new StringBuilder();

            foreach (var filePath in files)
            {
               var peFile = new PeNet.PeFile(filePath);
                
               var imports = peFile.ImageResourceDirectory.NumberOfIdEntries;
               var sizeOfData = peFile.ImageSectionHeaders.FirstOrDefault()?.SizeOfRawData;

               var threatClassification = string.Empty;

               foreach (var name in Enum.GetNames(typeof(ThreatTypes)))
               {
                   if (!filePath.Contains(name))
                   {
                       continue;
                   }

                   threatClassification = name;
               }

               if (string.IsNullOrEmpty(threatClassification))
               {
                   Console.WriteLine($"{filePath} was not named properly");

                   continue;                   
               }

               sb.AppendLine($"{threatClassification},{imports},{sizeOfData}");
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
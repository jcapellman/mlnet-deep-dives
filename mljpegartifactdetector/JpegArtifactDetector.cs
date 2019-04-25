using System;
using System.IO;
using System.Text;

using mldeepdivelib.Abstractions;
using mldeepdivelib.Enums;
using mldeepdivelib.Helpers;

using mljpegartifactdetector.Structures;

using Microsoft.ML;
using Microsoft.ML.Transforms;

namespace mljpegartifactdetector
{
    public class JpegArtifactDetector : BaseMLPrediction
    {
        protected override void FeatureExtraction(string[] args)
        {
            var startDate = DateTime.Now;

            var files = Directory.GetFiles(args[(int)CommandLineArguments.INPUT_FILE]);

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

            File.WriteAllText(args[(int)CommandLineArguments.OUTPUT_FILE], sb.ToString());

            Console.WriteLine($"Feature Extraction completed in {DateTime.Now.Subtract(startDate).TotalMinutes} minutes to {args[(int)CommandLineArguments.OUTPUT_FILE]}");
        }
        
        protected override void Train(string[] args)
        {
            var trainingDataView = MlContext.Data.LoadFromTextFile<JpegArtifactorDetectorData>(args[(int)CommandLineArguments.INPUT_FILE], hasHeader: true);

            var dataProcessPipeline = MlContext.Transforms.Conversion.MapValueToKey("Label", "Number", 
                    keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue).
                Append(MlContext.Transforms.Concatenate("Features", "DataFeaturized").AppendCacheCheckpoint(MlContext));

            var trainer = MlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Data");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            var trainedModel = trainingPipeline.Fit(trainingDataView);

            using (var fs = File.Create(args[(int)CommandLineArguments.OUTPUT_FILE]))
            {
                MlContext.Model.Save(trainedModel, trainingDataView.Schema, fs);
            }

            Console.WriteLine($"Saved model to {args[(int)CommandLineArguments.OUTPUT_FILE]}");
        }

        protected override void Predict(string[] args)
        {
            var prediction = Predictor.Predict<JpegArtifactorDetectorData, JpegArtifactorDetectorPrediction>(MlContext, args[(int)CommandLineArguments.INPUT_FILE], args[(int)CommandLineArguments.OUTPUT_FILE]);

            Console.WriteLine($"Has Jpeg Artifacts: {prediction.ContainsJpegArtifacts:0.#}");
        }

        private JpegArtifactorDetectorData FeatureExtractFile(string filePath, bool forPrediction = false)
        {
            return new JpegArtifactorDetectorData
            {
                FilePath = filePath
            };
        }
    }
}
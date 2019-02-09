using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Text;

using mldeepdivelib.Abstractions;
using mldeepdivelib.Enums;
using mldeepdivelib.Helpers;

using mljpegartifactdetector.Structures;

using Microsoft.ML;
using Microsoft.ML.Data;

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
            var trainingDataView = MlContext.Data.ReadFromTextFile<JpegArtifactorDetectorData>(args[(int)CommandLineArguments.INPUT_FILE], hasHeader: false, separatorChar: ',');

            var dataProcessPipeline = MlContext.Transforms.Conversion.MapValueToKey(
                    outputColumnName: DefaultColumnNames.Label,
                    inputColumnName: nameof(JpegArtifactorDetectorData.ContainsJpegArtifacts))
                .Append(MlContext.Transforms.Text.FeaturizeText(outputColumnName: "DataFeaturized",
                    inputColumnName: nameof(JpegArtifactorDetectorData.Data)))
                .Append(MlContext.Transforms.Concatenate(DefaultColumnNames.Features,"DataFeaturized"));
            
            var trainer = MlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumn: "Label", featureColumn: "Data");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            var trainedModel = trainingPipeline.Fit(trainingDataView);

            using (var fs = File.Create(args[(int)CommandLineArguments.OUTPUT_FILE]))
            {
                trainedModel.SaveTo(MlContext, fs);
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
            using (var image = new Bitmap(Image.FromFile(filePath)))
            {
                var data = new List<int>();

                for (var x = 0; x < image.Width; x++)
                {
                    for (var y = 0; y < image.Height; y++)
                    {
                        data.Add(image.GetPixel(x, y).ToArgb());
                    }
                }
                
                return new JpegArtifactorDetectorData
                {
                    Data = data.ToArray()
                };
            }
        }
    }
}
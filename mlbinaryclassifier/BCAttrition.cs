using System;
using System.IO;

using mlbinaryclassifier.Structures;

using mldeepdivelib.Abstractions;
using mldeepdivelib.Helpers;

using Microsoft.ML;
using Microsoft.ML.Data;

namespace mlbinaryclassifier
{
    public class BCAttrition : BaseMLPrediction
    {
        protected override void Train(string[] args)
        {
            var modelObject = Activator.CreateInstance<BCData>();

            var textReader = MlContext.Data.CreateTextReader(new TextLoader.Arguments
            {
                Separator = ",",
                HasHeader = false,
                Column = new[]
                {
                    new TextLoader.Column("StillThere", DataKind.R4, 0),
                    new TextLoader.Column("LongCommute", DataKind.R4, 1),
                    new TextLoader.Column("PromotionLimited", DataKind.R4, 2),
                    new TextLoader.Column("Overwhelmed", DataKind.R4, 3),
                    new TextLoader.Column("Label", DataKind.Text, 4)
                }
            });

            var baseTrainingDataView = textReader.Read(args[1]);

            var pipeline = MlContext.Transforms.Concatenate("Features", "StillThere", "LongCommute", "PromotionLimited", "Overwhelmed")
                .Append(MlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 5));

            var trainedModel = pipeline.Fit(baseTrainingDataView);

            using (var fs = File.Create(args[2]))
            {
                trainedModel.SaveTo(MlContext, fs);
            }

            Console.WriteLine($"Saved model to {args[2]}");
        }

        protected override void Predict(string[] args)
        {
            var prediction = Predictor.Predict<BCData, BCPrediction>(MlContext, args[1], args[2]);

            Console.WriteLine($"% Likely going to stay: {prediction.Probability:0.#}");
        }

        protected override void FeatureExtraction(string[] args)
        {
            throw new NotImplementedException();
        }
    }
}
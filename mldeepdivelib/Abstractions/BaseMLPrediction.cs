using System;
using System.IO;
using mldeepdivelib.Common;
using mldeepdivelib.Enums;

using Microsoft.ML;

namespace mldeepdivelib.Abstractions
{
    public abstract class BaseMLPrediction
    {
        protected MLContext MlContext;

        protected BaseMLPrediction()
        {
            MlContext = new MLContext(seed: 0);
        }

        protected abstract void Train(string[] args);

        protected abstract void Predict(string[] args);

        protected abstract void FeatureExtraction(string[] args);

        private CommandLineResponse parseCommandLine(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine($"No parameters passed to Prediction {System.Environment.NewLine}" +
                                  $"usage: <operation> <arg1> <arg2>");

                return new CommandLineResponse {Success = false};
            }

            if (!Enum.TryParse(typeof(MLOperations), args[0], out var mlOperation))
            {
                Console.WriteLine($"{args[0]} is an invalid argument");

                Console.WriteLine("Available Options:");

                Console.WriteLine(string.Join(", ", Enum.GetNames(typeof(MLOperations))));

                return new CommandLineResponse { Success = false };
            }

            if (args.Length < 3)
            {
                Console.WriteLine("Each Operation requires 2 additional arguments");
                
                return new CommandLineResponse { Success = false };
            }

            return new CommandLineResponse
            {
                Success = true,
                MLOperation = (MLOperations) mlOperation
            };
        }

        public void Run(string[] args)
        {
            Console.Clear();

            var response = parseCommandLine(args);

            if (!response.Success)
            {
                return;
            }

            switch (response.MLOperation)
            {
                case MLOperations.train:
                    Train(args);
                    break;
                case MLOperations.predict:
                    Predict(args);
                    break;
                case MLOperations.featureextraction:
                    FeatureExtraction(args);
                    break;
            }
        }
    }
}
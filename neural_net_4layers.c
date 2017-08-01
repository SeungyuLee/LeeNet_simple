#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <math.h>
#include "MNIST/mnist.h"

#define NumInput 784
#define NumHidden 100
#define NumHidden2 30
#define NumOutput 10
#define eta 0.1
#define alpha 0 // momentum value

double gaussian_rand(void);

double Input[NumInput+1];
double Hidden[NumHidden+1];
double Output[NumOutput+1];
double Target[NumOutput+1];
double SumHidden[NumHidden+1];
double SumOutput[NumOutput+1];
double WeightIH[NumInput+1][NumHidden+1];
double WeightHO[NumHidden2+1][NumOutput+1];
double BiasIH[NumHidden];
double BiasHO[NumOutput];
double DeltaWeightIH[NumInput+1][NumHidden+1];
double DeltaWeightHO[NumHidden2+1][NumOutput+1];
double DeltaOutput[NumOutput+1];
double DeltaHidden[NumHidden+1];
double SumDOW[NumHidden+1];
double Error;
int i, j, k;
int cnt = 0;

double Hidden2[NumHidden2+1];
double SumHidden2[NumHidden2+1];
double WeightHH[NumHidden+1][NumHidden2+1];
double DeltaWeightHH[NumHidden+1][NumHidden2+1];
double DeltaHidden2[NumHidden2+1];
double SumDOW2[NumHidden2+1];

int main(void){

	mnist_data *train_data;
	mnist_data *test_data;
	unsigned int train_cnt;
	unsigned int test_cnt;

	mnist_load("MNIST/train-images-idx3-ubyte", 
		"MNIST/train-labels-idx1-ubyte", &train_data, &train_cnt);

	mnist_load("MNIST/t10k-images-idx3-ubyte", 
		"MNIST/t10k-labels-idx1-ubyte", &test_data, &test_cnt);

	for(j = 1; j <= NumHidden; j++){
		for(i = 0; i <= NumInput; i++){
			DeltaWeightIH[i][j] = 0.0;
			WeightIH[i][j]= gaussian_rand();
		}
	}

	for(j = 1; j <= NumHidden2; j++){
		for(i = 0; i <= NumHidden; i++){
			DeltaWeightHH[i][j] = 0.0;
			WeightHH[i][j] = gaussian_rand();
		}
	}

	for(k = 1; k <= NumOutput; k++){
		for(j = 0; j <= NumHidden2; j++){
			DeltaWeightHO[j][k] = 0.0;
			WeightHO[j][k] = gaussian_rand();
		}
	}


	while( cnt < train_cnt ){	
		for(j = 0; j < 28; j++) // Input Setting
			for(k = 1; k <= 28; k++)
				Input[j*28+k] = train_data[cnt].data[j][k-1];

		for(i = 1; i <= NumOutput; i++){
			Target[i] = 0.0;
		}
		Target[train_data[cnt++].label] = 1.0;
		
		Error = 0.0;

		for(j = 1; j <= NumHidden; j++){ /* compute hidden unit activations */
			SumHidden[j] = WeightIH[0][j];
			for(i = 1; i <= NumInput; i++){
				SumHidden[j] += Input[i] * WeightIH[i][j];
			}
			Hidden[j] = 1.0/(1.0+exp(-SumHidden[j])); 
		}

		for(j = 1; j <= NumHidden2; j++){
			SumHidden2[j] = WeightHH[0][j];
			for(i = 1; i <= NumHidden; i++)
				SumHidden2[j] += Hidden[i] * WeightHH[i][j];
			Hidden2[j] = 1.0/(1.0+exp(-SumHidden2[j]));
		}

		for(k = 1; k <= NumOutput; k++){ /* compute output unit activations and errors */
			SumOutput[k] = WeightHO[0][k];
			for(j = 1; j <= NumHidden2; j++){
				SumOutput[k] += Hidden2[j] * WeightHO[j][k];
			}
			Output[k] = 1.0/(1.0+exp(-SumOutput[k]));
			Error += 0.5 * (Target[k] - Output[k]) * (Target[k] - Output[k]);
			DeltaOutput[k] = (Target[k] - Output[k]) * Output[k] * (1.0-Output[k]);
		}

		for(j = 1; j <= NumHidden2; j++){ /* back-propagate errors to hidden layer */
			SumDOW2[j] = 0.0;
			for(k = 1; k <= NumOutput; k++){
				SumDOW2[j] += WeightHO[j][k] * DeltaOutput[k];	
			}
			DeltaHidden2[j] = SumDOW2[j] * Hidden2[j] * (1.0 - Hidden2[j]);
		}

		for(j = 1; j <= NumHidden; j++){
			SumDOW[j] = 0.0;
			for(k = 1; k <= NumHidden2; k++)
				SumDOW[j] += WeightHH[j][k] * DeltaHidden2[k];
			DeltaHidden[j] = SumDOW[j] * Hidden[j] * (1.0 - Hidden[j]);
		}

		for(j = 1; j <= NumHidden; j++){ /* update weights WeightIH */
			DeltaWeightIH[0][j] = eta * DeltaHidden[j] + alpha * DeltaWeightIH[0][j];
			WeightIH[0][j] += DeltaWeightIH[0][j];
			for(i = 1; i <= NumInput; i++){
				DeltaWeightIH[i][j] = eta * Input[i] * DeltaHidden[j] + alpha * DeltaWeightIH[i][j];
				WeightIH[i][j] += DeltaWeightIH[i][j];
			}
		}

		for(j = 1; j <= NumHidden2; j++){
			DeltaWeightHH[0][j] = eta * DeltaHidden2[j] + alpha * DeltaWeightHH[0][j];
			WeightHH[0][j] += DeltaWeightHH[0][j];
			for(i = 1; i <= NumHidden; i++){
				DeltaWeightHH[i][j] = eta * Hidden[i] * DeltaHidden2[j] + alpha * DeltaWeightHH[i][j];
				WeightHH[i][j] += DeltaWeightHH[i][j];
			}
		}

		for(k = 1; k <= NumOutput; k++){
			DeltaWeightHO[0][k] = eta * DeltaOutput[k] + alpha * DeltaWeightHO[0][k];
			WeightHO[0][k] += DeltaWeightHO[0][k];
			for(j = 1; j <= NumHidden2; j++){
				DeltaWeightHO[j][k] = eta * Hidden2[j] * DeltaOutput[k] + alpha * DeltaWeightHO[j][k];
				WeightHO[j][k] += DeltaWeightHO[j][k];
			}	
		}

		if(cnt % 1000 == 0){	
			int correct_num = 0;
	
			printf("cnt is %d, Error value is %lf.\n", cnt, Error);
		
			for(i = 0; i < 10000; i++){
				for(j = 0; j < 28; j++){
					for(k = 1; k <= 28; k++){
						Input[j*28+k] = test_data[i].data[j][k-1];
					}
				}	
				
				for(j = 1; j <= NumHidden; j++){ 
					SumHidden[j] = WeightIH[0][j];
					for(k = 1; k <= NumInput; k++){
						SumHidden[j] += Input[k] * WeightIH[k][j];
					}
					Hidden[j] = 1.0/(1.0+exp(-SumHidden[j])); 
				}
				
				for(j = 1; j <= NumHidden2; j++){
					SumHidden2[j] = WeightHH[0][j];
					for(k = 1; k <= NumHidden; k++)
						SumHidden2[j] += Hidden[k] * WeightHH[k][j];
					Hidden2[j] = 1.0/(1.0+exp(-SumHidden2[j]));
				}

				for(k = 1; k <= NumOutput; k++){ 
					SumOutput[k] = WeightHO[0][k];
					for(j = 1; j <= NumHidden2; j++){
						SumOutput[k] += Hidden2[j] * WeightHO[j][k];
					}
					Output[k] = 1.0/(1.0+exp(-SumOutput[k]));
				}

				int max = 0;
				double max_result = 0;
				for(k = 1; k <= 10; k++){
					if(max_result < Output[k]) { max_result = Output[k]; max = k; }
				}
				if(max == test_data[i].label) correct_num++;
			}
	
			printf("correct number: %d / 10000\n", correct_num);
			
	}
	
	
	}
	return 0;
}

time_t rseed = 0;
#define PI 3.141592654
double gaussian_rand(void)
{
	static double U, V;
	static int phase = 0;
	double Z;

	if(rseed==0){
		time(&rseed);
		srand(rseed);
	}	
	
	if(phase == 0){
			U = (rand() + 1.) / (RAND_MAX + 2.);
			V = rand() / (RAND_MAX + 1.);
			Z = sqrt(-2 * log(U)) * sin(2*PI*V);
	}else
			Z = sqrt(-2 * log(U)) * cos(2*PI*V);

	phase = 1 - phase;
	return Z;
}

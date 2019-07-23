import java.util.Random;

/**
 * 분류를 위한 MultiLayerPerceptron
 * 입력층 개수는 데이터 개수이고 출력층 개수는 분류할 항목들의 개수로 설정하면 된다.
 * 
 * 기존 MLP와의 차이점은 오차율을 바탕으로 학습률을 업데이트 한다.
 * 새로운 학습률 = 초기학습률 * 오차율
 * 
 * @author Hwiyong Chang
 */
public class MLP {
	private static final double RANDOM_RANGE_ABSOLUTE_VALUE = 1;	// 랜덤값의  절대값 범위
	
	private int layerCount;					// 층의 개수
	private int[] neuralCount;				// 각 층의 뉴런 개수
	private double initialLearningRate;		// 초기 학습률(고정)
	private double learningRate;			// 학습률
	private Perceptron[][] neuralNetwork;	// 중간층과 출력층만을 나타낸 NN
	
	/**
	 * MLP 생성자 : 중간층과 출력층만 퍼셉트론이 존재한다.
	 * @param neuralCount 각 층의 뉴런 개수 배열
	 * @param initialLearningRate 초기 학습률
	 */
	public MLP(int[] neuralCount, double initialLearningRate) {
		this.layerCount = neuralCount.length;
		this.neuralCount = neuralCount;
		this.initialLearningRate = this.learningRate = initialLearningRate;
		this.neuralNetwork = new Perceptron[layerCount-1][];
		initNeuralNetwork();
		randomThreshold();
		randomWeight();
	}
	
	/**
	 * NN 생성 및 초기화 메소드
	 * N층의 뉴런들이 각자 (N-1)층의 뉴런 개수만큼 가중치를 가지고 있어야 한다. 
	 */
	private void initNeuralNetwork() {
		for(int layerIndex = 1; layerIndex < layerCount; ++layerIndex) {
			neuralNetwork[layerIndex-1] = new Perceptron[neuralCount[layerIndex]];
			for(int perceptronIndex = 0; perceptronIndex < neuralCount[layerIndex]; ++perceptronIndex)
				neuralNetwork[layerIndex-1][perceptronIndex] = new Perceptron(neuralCount[layerIndex-1]);
		}
	}

	/**
	 * 임계값을 랜덤값으로 초기화하는 메소드
	 */
	private void randomThreshold() {
		for(int layerIndex = 1; layerIndex < layerCount; ++layerIndex)
			for(int perceptronIndex = 0; perceptronIndex < neuralCount[layerIndex]; ++perceptronIndex)
				neuralNetwork[layerIndex-1][perceptronIndex].setThreshold(getDoubleTypeRandom(-RANDOM_RANGE_ABSOLUTE_VALUE, RANDOM_RANGE_ABSOLUTE_VALUE));
	}

	/**
	 * 가중치 값을 랜덤값으로 초기화하는 메소드
	 */
	private void randomWeight() {
		for(int layerIndex = 1; layerIndex < layerCount; ++layerIndex)
			for(int perceptronIndex = 0; perceptronIndex < neuralCount[layerIndex]; ++perceptronIndex)
				for(int weightIndex = 0; weightIndex < neuralCount[layerIndex-1]; ++weightIndex)
					neuralNetwork[layerIndex-1][perceptronIndex].setWeight(weightIndex, getDoubleTypeRandom(-RANDOM_RANGE_ABSOLUTE_VALUE, RANDOM_RANGE_ABSOLUTE_VALUE));
	}
	
	/**
	 * 정해진 범위의 값 사이에서 임의의 값을 반환하는 메소드
	 * @param startValue 시작범위 값
	 * @param endValue 끝범위 값
	 * @return 랜덤 값
	 */
	private double getDoubleTypeRandom(double startValue, double endValue) {
		return Math.random() * (endValue - startValue) + startValue;
	}
	
	/**
	 * 입력층 데이터 목록과 해당 데이터에 맞는 정답을 받아서 훈련횟수만큼 훈련시키는 메소드
	 * (epoch 메소드를 trainCount번 실행)
	 * @param trainCount 훈련시킬 횟수
	 * @param trainInputDataList 입력층 데이터 목록
	 * @param trainAnswerIndexList 데이터 정답 인덱스 목록
	 * @return epoch훈련의 각 항목과 전체에 대한 정답률이 기록된 정확도 정보 반환[훈련횟수][출력층 개수+1][3]
	 */
	public double[][][] train(int trainCount, double[][] trainInputDataList, int[] trainAnswerIndexList) {
		double[][][] accuracyEpochList = new double[trainCount][][];
		for(int trainIndex = 0; trainIndex < trainCount; ++trainIndex)
			accuracyEpochList[trainIndex] = epoch(trainInputDataList, trainAnswerIndexList, true);
		return accuracyEpochList;
	}
	
	/**
	 * 입력층 데이터 목록과 해당 데이터에 맞는 정답을 받아서 1번 훈련시키는 메소드
	 * 훈련이라면 역전파 메소드까지 싱행하여 임계값과 가중치값을 갱신한다.
	 * 마지막에는 학습률도 갱신한다.
	 * @param inputDataList 입력층 데이터 목록
	 * @param answerIndexList 데이터 정답 인덱스 목록
	 * @param isTrain 훈련인지 테스트인지 구별
	 * @return 각 항목과 전체에 대한 정답률이 기록된 정확도 정보 반환[출력층 개수+1][3]
	 */
	private double[][] epoch(double[][] inputDataList, int[] answerIndexList, boolean isTrain) {
		// correctList : 각 분류항목에 대한 {맞은 개수, 항목 개수}를 계산
		// 마지막 인덱스에는 총 개수에 대한 {맞은 개수, 항목 개수}를 계산
		int[][] correctList = new int[neuralCount[layerCount-1]][2];
		int trainDataCount = answerIndexList.length;
		for(int dataIndex = 0; dataIndex < trainDataCount; ++dataIndex) {
			double[][] outputList = frontPropagation(inputDataList[dataIndex]);
			++correctList[answerIndexList[dataIndex]][1];
			if(answerIndexList[dataIndex] == getMaxIndex(outputList[layerCount-1]))
				++correctList[answerIndexList[dataIndex]][0];
			if(isTrain)		// 훈련이라면, 임계값과 가중치값들을 갱신하는 역전파 부분을 실행
				backPropagation(outputList, answerIndexList[dataIndex]);
		}
		double[][] accuracyList = getAccuracyList(correctList);
		learningRateUpdate(1 - accuracyList[neuralCount[layerCount-1]][2]);		// 학습률 갱신
		return accuracyList;
	}
	
	/**
	 * 순전파 메소드 : 각 층의 출력값들을 모아서 반환
	 * @param inputData 입력층 데이터
	 * @return 모든 층들의 결과목록
	 */
	private double[][] frontPropagation(double[] inputData) {
		double[][] output = new double[layerCount][];
		for(int layerIndex = 0; layerIndex < layerCount; ++layerIndex) {
			output[layerIndex] = new double[neuralCount[layerIndex]];
			for(int perceptronIndex = 0; perceptronIndex < neuralCount[layerIndex]; ++perceptronIndex) {
				if(layerIndex == 0)				// 입력층 : inputData값을 그대로 사용
					output[layerIndex][perceptronIndex] = inputData[perceptronIndex];
				else {
					double sum = -neuralNetwork[layerIndex-1][perceptronIndex].getThreshold();	// sum = -threshold;
					// perceptronIndex가 N층의 색인이라면, weightIndex는 (N-1)층의 색인이다.
					for(int weightIndex = 0; weightIndex < neuralCount[layerIndex-1]; ++weightIndex) {
						if(layerIndex == 1)		// 1번째 중간층 : sum += w * x(입력층)
							sum += neuralNetwork[layerIndex-1][perceptronIndex].getWeight(weightIndex) * inputData[weightIndex];
						else					// 그 이외 : sum += w * x(중간층)
							sum += neuralNetwork[layerIndex-1][perceptronIndex].getWeight(weightIndex) * output[layerIndex-1][weightIndex];
					}
					output[layerIndex][perceptronIndex] = sigmoid(sum);
				}
			}
		}
		return output;
	}
	
	/**
	 * 역전파 메소드 : 출력층부터 시작하여 각 층의 가중치값과 임계값을 갱신
	 * @param outputList 순전파 메소드에서 출력된 모든 층들의 결과목록
	 * @param answerIndex 해당 입력층 데이터의 정답 인덱스
	 */
	private void backPropagation(double[][] outputList, double answerIndex) {
		// 입력층은 오차기울기가 없으므로 중간층들과 출력층만을 고려하여 배열 길이를 정함
		double[][] errorGradient = new double[layerCount-1][];	// 오차기울기 배열
		for(int layerIndex = layerCount-1; layerIndex > 0; --layerIndex) {
			errorGradient[layerIndex-1] = new double[neuralCount[layerIndex]];
			for(int perceptronIndex = 0; perceptronIndex < neuralCount[layerIndex]; ++perceptronIndex) {
				double output = outputList[layerIndex][perceptronIndex];
				if(layerIndex == layerCount-1) {	// 출력층
					
					// 출력충의 오차 기울기 구하기 (오차기울기 = 오차 * 뉴런값의 편미분)
					double error = (perceptronIndex == answerIndex ? 1 : 0) - output;	// 분류 알고리즘의 오차
					errorGradient[layerIndex-1][perceptronIndex] = error * sigmoidPartialDerivative(output);
					
					// 출력층 뉴런들의 가중치 갱신 (다음 가중치 = 현재 가중치 + 학습률 * 연결된 마지막 은닉층의 뉴런값 * 해당 오차기울기)
					// perceptronIndex가 N층의 색인이라면, weightIndex는 (N-1)층의 색인이다.
					for(int weightIndex = 0; weightIndex < neuralCount[layerIndex-1]; ++weightIndex)
						neuralNetwork[layerIndex-1][perceptronIndex].addWeight(weightIndex, learningRate * outputList[layerIndex-1][weightIndex] * errorGradient[layerIndex-1][perceptronIndex]);
					
					// 출력층 뉴런의 임계값 갱신 (다음 임계값 = 현재 임계값 + 학습률 * -1 * 해당 오차기울기)
					neuralNetwork[layerIndex-1][perceptronIndex].addThreshold(learningRate * -1 * errorGradient[layerIndex-1][perceptronIndex]);
				}
				else {								// 중간층
					
					// 은닉층의 오차 기울기 구하기 (오차기울기 = 뉴런값의 편미분 * sum(다음층 오차기울기 * 현재 은닉층에 연결된 가중치))
					double hiddenLayerErrorSum = 0.0;
					for(int nextLayerPerceptronIndex = 0; nextLayerPerceptronIndex < neuralCount[layerIndex+1]; ++nextLayerPerceptronIndex)
						hiddenLayerErrorSum += errorGradient[layerIndex][nextLayerPerceptronIndex] * neuralNetwork[layerIndex][nextLayerPerceptronIndex].getWeight(perceptronIndex);
					errorGradient[layerIndex-1][perceptronIndex] = sigmoidPartialDerivative(output) * hiddenLayerErrorSum;
					
					// 은닉층 뉴련들의 가중치 갱신 (다음 가중치 = 현재 가중치 + 학습률 * 연결된 이전층의 뉴런값 * 해당 오차기울기)
					for(int weightIndex = 0; weightIndex < neuralCount[layerIndex-1]; ++weightIndex)
						neuralNetwork[layerIndex-1][perceptronIndex].addWeight(weightIndex, learningRate * outputList[layerIndex-1][weightIndex] * errorGradient[layerIndex-1][perceptronIndex]);
					
					// 은닉층 뉴런의 임계값 갱신 (다음 임계값 = 현재 임계값 + 학습률 * -1 * 해당 오차기울기)
					neuralNetwork[layerIndex-1][perceptronIndex].addThreshold(learningRate * -1 * errorGradient[layerIndex-1][perceptronIndex]);
				}
			}
		}
	}
	
	/**
	 * 새로운 double[][3]의 2차원 배열을 만든 후,
	 * [0]=맞은 개수, [1]=항목 개수 목록인 correctList를 받아서 복사하고 [2]에 정확도로 변환한다.
	 * Row를 하나 더 추가하여 마지막 Row에는 총 맞은 개수, 총 항목 개수, 정확도를 기록한다.  
	 * @param correctList [0]은 맞은개수, [1]은 전체 개수가 기록된 [][2]배열
	 * @return 정확도 정보가 담긴 double[][3]타입의 2차원 배열
	 */
	private double[][] getAccuracyList(int[][] correctList) {
		// accuracyList : 각 분류항목에 대한 {맞은 개수, 항목 개수, 정확도}를 계산
		// 마지막 인덱스에는 총 개수에 대한 {전체 맞은 개수, 전체 항목 개수, 전체 정확도}를 계산
		double[][] accuracyList = new double[neuralCount[layerCount-1]+1][3];
		for(int outputIndex = 0; outputIndex < neuralCount[layerCount-1]+1; ++outputIndex) {
			if(outputIndex < neuralCount[layerCount-1]) {
				accuracyList[outputIndex][0] = correctList[outputIndex][0];
				accuracyList[outputIndex][1] = correctList[outputIndex][1];
				accuracyList[neuralCount[layerCount-1]][0] += correctList[outputIndex][0];
				accuracyList[neuralCount[layerCount-1]][1] += correctList[outputIndex][1];
			}
			accuracyList[outputIndex][2] = accuracyList[outputIndex][0] / accuracyList[outputIndex][1];
		}
		return accuracyList;
	}
	
	/**
	 * 학습률을 갱신하는 메소드
	 * 학습이 잘 될수록 학습률을 낮춰야한다.
	 * 새로운 학습률 = 초기 학습률 * 오차율
	 * @param errorRate 오차율
	 */
	private void learningRateUpdate(double errorRate) {
		learningRate = initialLearningRate * errorRate;
	}
	
	/**
	 * 입력층 데이터 목록과 해당 데이터에 맞는 정답을 받아서 정답률을 시험하는 메소드
	 * @param testInputDataList 입력층 데이터 목록
	 * @param testAnswerIndexList 데이터 정답 인덱스 목록
	 * @return 각 항목과 전체에 대한 정답률이 기록된 정확도 정보 반환[출력층 개수+1][3]
	 */
	public double[][] test(double[][] testInputDataList, int[] testAnswerIndexList) {
		return epoch(testInputDataList, testAnswerIndexList, false);
	}
	
	/**
	 * 입력데이터 목록을 받아서 각 데이터마다 분류에 대한 각 항목의 예측 확률 계산
	 * 각 예측값의 합이 1이 되도록 변환
	 * @param inputDataList 입력층 데이터 목록
	 * @return 데이터마다 각 항목에 대한 확률 정보가 기록된 2차원 배열 
	 */
	public double[][] predict(double[][] inputDataList) {
		double[][] probabilityList = new double[inputDataList.length][];
		for(int listIndex = 0; listIndex < inputDataList.length; ++listIndex)
			probabilityList[listIndex] = predict(inputDataList[listIndex]);
		return probabilityList;
	}
	
	/**
	 * 입력데이터를 받아서 분류에 대한 각 항목의 예측 확률 계산
	 * 각 예측값의 합이 1이 되도록 변환
	 * @param inputData 입력층 데이터
	 * @return 각 항목에 대한 확률 배열
	 */
	public double[] predict(double[] inputData) {
		double[] output = frontPropagation(inputData)[layerCount-1];
		// output[outputIndex] : resultSum = probability : 1.0
		// probability = output[outputIndex] / resultSum
		double resultSum = 0.0;
		for(int outputIndex = 0; outputIndex < output.length; ++outputIndex)
			resultSum += output[outputIndex];
		for(int outputIndex = 0; outputIndex < output.length; ++outputIndex)
			output[outputIndex] /= resultSum;
		return output;
	}
	
	/**
	 * 배열에서 최고값을 반환하는 함수
	 * @param array 배열
	 * @return 배열의 최고값
	 */
	private static int getMaxIndex(double[] array) {
		int maxValueIndex = 0;
		for(int valueIndex = 0; valueIndex < array.length; ++valueIndex)
			if(array[valueIndex] > array[maxValueIndex])
				maxValueIndex = valueIndex;
		return maxValueIndex;
	}
	
	/**
	 * 시그모이드 함수
	 * @param x 입력값
	 * @return 1 / (1 + exponential(-x))
	 */
	private static double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	/**
	 * 시그모이드 편도함수
	 * @param y sigmoid(x)
	 * @return (d/dx)sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
	 */
	private static double sigmoidPartialDerivative(double y) {
		return y * (1 - y);
	}
	
	/**
	 * inputDataList를 rate비율로 나눠주는 함수
	 * inputDataList가 n개의 Row를 가지면, inputSplit[0]의 길이는 (rate * n) 이고, inputSplit[1]의 길이는 (n - inputSplit[0].length)
	 * @param inputDataList 입력층 데이터 목록
	 * @param rate 나누는 비율
	 * @param seed 랜덤에 사용할 시드
	 * @param shuffle 섞기 여부
	 * @return 입력층 데이터 목록을 rate비율로 나눈 3차원 입력층 데이터 목록
	 */
	public static double[][][] splitInputDataList(double[][] inputDataList, double rate, int seed, boolean shuffle) {
		int listCount = inputDataList.length;
		if(shuffle) {
			Random rand = new Random(seed);
			for(int index1 = 0; index1 < listCount; ++index1) {
				int index2 = (int)(rand.nextInt(listCount));
				swapDoubleArray(inputDataList, index1, index2);
			}
		}
		double[][][] inputSplit = new double[2][][];
		inputSplit[0] = new double[(int)(rate * listCount)][];
		inputSplit[1] = new double[listCount - inputSplit[0].length][];
		for(int i=0; i<listCount; ++i) {
			if(i < inputSplit[0].length)
				inputSplit[0][i] = inputDataList[i];
			else
				inputSplit[1][i - inputSplit[0].length] = inputDataList[i];
		}
		return inputSplit;
	}
	
	/**
	 * 2차원 Double타입 배열에서 index1, index2에 해당하는 배열을 바꾸는 함수
	 * @param twoDimensionalArray
	 * @param index1
	 * @param index2
	 */
	private static void swapDoubleArray(double[][] twoDimensionalArray, int index1, int index2) {
		double[] temp = twoDimensionalArray[index1];
		twoDimensionalArray[index1] = twoDimensionalArray[index2];
		twoDimensionalArray[index2] = temp;
	}
	
	/**
	 * answerIndexList를 rate비율로 나눠주는 함수
	 * answerIndexList가 n개의 Row를 가지면, split[0]의 길이는 (rate * n) 이고, split[1]의 길이는 (n - split[0].length)
	 * @param answerIndexList 정답 인덱스 목록
	 * @param rate 나누는 비율
	 * @param seed 랜덤에 사용할 시드
	 * @param shuffle 섞기 여부
	 * @return 정답 인덱스 목록을 rate비율로 나눈 2차원 정답 인덱스 목록
	 */
	public static int[][] splitAnswerIndexList(int[] answerIndexList, double rate, int seed, boolean shuffle) {
		int listCount = answerIndexList.length;
		if(shuffle) {
			Random rand = new Random(seed);
			for(int index1 = 0; index1 < listCount; ++index1) {
				int index2 = (int)(rand.nextInt(listCount));
				swapInteger(answerIndexList, index1, index2);
			}
		}
		int[][] answerSplit = new int[2][];
		answerSplit[0] = new int[(int)(rate * listCount)];
		answerSplit[1] = new int[listCount - answerSplit[0].length];
		for(int i=0; i<listCount; ++i) {
			if(i < answerSplit[0].length)
				answerSplit[0][i] = answerIndexList[i];
			else
				answerSplit[1][i - answerSplit[0].length] = answerIndexList[i];
		}
		return answerSplit;
	}
	
	/**
	 * 정수 배열에서 index1, index2에 해당하는 값을 바꾸는 함수
	 * @param integerArray
	 * @param index1
	 * @param index2
	 */
	private static void swapInteger(int[] integerArray, int index1, int index2) {
		int temp = integerArray[index1];
		integerArray[index1] = integerArray[index2];
		integerArray[index2] = temp;
	}
}

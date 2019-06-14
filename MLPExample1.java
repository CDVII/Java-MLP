/**
 * 십진수 숫자를 이진수로 바꾸어서 MODULAR_COUNT에 대한 나머지 연산을 한다.
 * 해당 이진수 정보를 입력층으로 사용하여 MODULAR_COUNT에 대한 나머지를 잘 분류하는지 MLP를 사용한다.
 * (MODULAR_COUNT를 2의 거듭제곱값로 정하면 이진수의 특징때문에 학습이 매우 잘 되므로, MODULAR_COUNT를 2의 거듭제곱값으로 사용 지양)
 * 
 * 주의점
 * 1. NUMBER_LIMIT < 2^BINARY_DIGIT_LENGTH
 * 2. MODULAR_COUNT != 2^n 
 * 
 * 후기
 * NUMBER_LIMIT가 클수록(데이터 양이 많을수록) 학습이 잘 된다.
 * TRAIN_RATE가 클수록(훈련 데이터 양의 비중이 클수록) 학습이 잘 된다.
 * 
 * @author Hwiyong Chang
 */
public class MLPExample1 {
	
	// 데이터 상수 부분
	private static final int NUMBER_LIMIT = 1000;		// data로 사용할 숫자(1 ~ NUMBER_LIMIT)
	private static final int BINARY_DIGIT_LENGTH = 10;	// 무조건 NUMBER_LIMIT < 2^BINARY_DIGIT_LENGTH (=입력층 개수)
	private static final int MODULAR_COUNT = 10;		// 나머지 (=출력층 개수)
	
	// MLP 상수 부분
	private static final int[] NEURON_COUNT = {BINARY_DIGIT_LENGTH, 16, 20, MODULAR_COUNT};	// 뉴런층 개수
	private static final double INITIAL_LEARNING_RATE = 0.1;	// 초기 학습률
	private static final double TRAIN_RATE = 0.75;	// 총 데이터에서 훈련 데이터 비중
	private static final int SEED = 1;	// Random객체 시드
	private static final int TRAIN_COUNT = 10000;	// 훈련 횟수
	
	public static void main(String[] args) {
		// 데이터 생성
		int[] data = new int[NUMBER_LIMIT];
		for(int i=0; i<NUMBER_LIMIT; ++i)
			data[i] = i+1;
		
		// MLP 생성
		MLP mlp = new MLP(NEURON_COUNT, INITIAL_LEARNING_RATE);
		
		// 데이터를 MLP 입력층과 출력층에 맞도록
		// 입력데이터와 정답데이터로 변환
		double[][] inputDataList = getInputDataList(data);
		int[] answerIndexList = getAnswerIndexList(data);
		
		// 데이터와 정답 리스트를 훈련과 시험용으로 분배 (같은 SEED를 사용해야 입력과 정답의 인덱스가 일치함)
		double[][][] inputSplit = MLP.splitInputDataList(inputDataList, TRAIN_RATE, SEED, true);
		int[][] answerSplit = MLP.splitAnswerIndexList(answerIndexList, TRAIN_RATE, SEED, true);
		
		// [0] = 훈련용, [1] = 시험용
		double[][] trainInputDataList = inputSplit[0];
		int[] trainAnswerIndexList = answerSplit[0];
		double[][] testInputDataList = inputSplit[1];
		int[] testAnswerIndexList = answerSplit[1];		
		
		// 훈련 결과 출력
		double[][][] trainAccuracyList = mlp.train(TRAIN_COUNT, trainInputDataList, trainAnswerIndexList);
		System.out.println("Train");
		for(int i=0; i<trainAccuracyList.length; ++i) {
			System.out.println("epoch " + (i+1));
			accuracyListPrint(trainAccuracyList[i]);
		}
		
		// 시험 결과 출력
		double[][] testAccuracyList = mlp.test(testInputDataList, testAnswerIndexList);
		System.out.println("Test");
		accuracyListPrint(testAccuracyList);
		
		// 예측해보기
		System.out.println("Predict");
		double[] randomNumber = testInputDataList[(int)(Math.random() * testInputDataList.length)];
		double[] predict = mlp.predict(randomNumber);
		int value = 0;		// 값 계산
		for(int i=0; i<randomNumber.length-1; ++i) {
			value += randomNumber[i];
			value <<= 1;
		}
		value += randomNumber[randomNumber.length-1];
		System.out.println("Number " + value + "'s predict result.");
		for(int i=0; i<predict.length; ++i)
			System.out.printf("\tprobability %d = %.3f\n", i, predict[i]);
	}
	
	/**
	 * 숫자 목록을 이진수로 변환하여  0과 1의 상태로 반환하는 함수
	 * @param data 숫자 목록
	 * @return 각 십진수 숫자에 대해 이진수로 변환된 0과 1 정보가 기록된 배열들
	 */
	private static double[][] getInputDataList(int[] data) {
		double[][] inputDataList = new double[data.length][NEURON_COUNT[0]];
		for(int dataIndex = 0; dataIndex < data.length; ++dataIndex) {
			String binaryString = Integer.toBinaryString(data[dataIndex]).toString();
			int startIndex = NEURON_COUNT[0] - binaryString.length();
			for(int binaryIndex = startIndex; binaryIndex < NEURON_COUNT[0]; ++binaryIndex)
				inputDataList[dataIndex][binaryIndex] = binaryString.charAt(binaryIndex - startIndex) - '0';
		}
		return inputDataList;
	}
	
	/**
	 * 숫자 목록을 출력층의 뉴런개수에 맞춰 나머지 정보를 반환하는 함수
	 * @param data 숫자 목록
	 * @return 나머지 값이 기록된 배열들
	 */
	private static int[] getAnswerIndexList(int[] data) {
		int[] answerIndexList = new int[data.length];
		for(int dataIndex = 0; dataIndex < data.length; ++dataIndex)
			answerIndexList[dataIndex] = data[dataIndex] % NEURON_COUNT[NEURON_COUNT.length-1];
		return answerIndexList;
	}
	
	/**
	 * MLP에서 훈련이나 테스트를 시켰을 때 반환되는 정확도 정보를 출력하는 함수
	 * @param accuracyList MLP에서 반환된 정확도 정보 목록
	 */
	private static void accuracyListPrint(double[][] accuracyList) {
		int len = accuracyList.length;
		for(int i=0; i<len-1; ++i)
			System.out.println("\tAccuracy " + i + " = " + (int)accuracyList[i][0] + "/" + (int)accuracyList[i][1] + " = " + accuracyList[i][2]);
		System.out.println("\tTotal accuracy = " + (int)accuracyList[len-1][0] + "/" + (int)accuracyList[len-1][1] + " = " + accuracyList[len-1][2]);
	}
}

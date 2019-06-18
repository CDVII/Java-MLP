import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;

/**
 * 2019년 1학기 강원대학교 컴퓨터정보통신공학 인공지능 강의 과제1
 * 태아 심박동 데이터를 이용한 분석
 * 
 * 후기
 * 신경망의 초기 학습률에 따라서 정답률의 차이가 크다.
 * 
 * @author Hwiyong Chang
 */
public class MLPExample2 {
	
	// 데이터 상수 부분
	private static final int DATA_LIST_COUNT = 2126;	// 데이터 목록 개수
	private static final int INPUT_COUNT = 21;			// 데이터 종류 (=입력층 개수)
	private static final int OUTPUT_COUNT = 3;			// 분류 (=출력층 개수)

	// MLP 상수 부분
	private static final int[] NEURON_COUNT = {INPUT_COUNT, 40, OUTPUT_COUNT};	// 뉴런층 개수
	private static final double INITIAL_LEARNING_RATE = 0.01;	// 초기 학습률
	private static final double TRAIN_RATE = 0.8;	// 총 데이터에서 훈련 데이터 비중
	private static final int SEED = 1;	// Random객체 시드
	private static final int TRAIN_COUNT = 10000;	// 훈련 횟수
	
	// 데이터가 저장된 파일 부분
	private static final String file_path = "";
	private static final String data_file_name = "MLPExample2Data.csv";
	
	public static void main(String[] args) {
		// 입력층 데이터 목록과 정답 목록을 저장할 배열 생성
		double[][] inputDataList = new double[DATA_LIST_COUNT][];
		for(int i=0; i<DATA_LIST_COUNT; ++i)
			inputDataList[i] = new double[INPUT_COUNT];
		int[] answerIndexList = new int[DATA_LIST_COUNT];
		
		// 파일에서 데이터 가져오기
		readData(inputDataList, answerIndexList);
		
		// MLP 생성
		MLP mlp = new MLP(NEURON_COUNT, INITIAL_LEARNING_RATE);
		
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
		int randomIndex = (int)(Math.random() * testInputDataList.length);
		double[] randomNumber = testInputDataList[randomIndex];
		double[] predict = mlp.predict(randomNumber);
		System.out.println("Answer " + testAnswerIndexList[randomIndex] + "'s predict result.");
		for(int i=0; i<predict.length; ++i)
			System.out.printf("\tprobability %d = %.3f\n", i, predict[i]);
	}
	
	/**
	 * 파일에 있는 데이터를 입력층 데이터와 정답 데이터로 분류한다.
	 * @param inputDataList 입력층 데이터 목록
	 * @param answerIndexList 정답 인덱스 목록
	 */
	private static void readData(double[][] inputDataList, int[] answerIndexList) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(file_path + data_file_name));
			int dataListIndex = 0;
			String line;
			while(reader.ready()) {
				line = reader.readLine();
				StringTokenizer tokens = new StringTokenizer(line, ",");
				for(int i=0; i<INPUT_COUNT; ++i)
					inputDataList[dataListIndex][i] = Double.parseDouble(tokens.nextToken());
				tokens.nextToken();		// 필요없는 데이터는 건너뛴다.
				// 정답 데이터는 인덱스에 맞게 저장한다.
				answerIndexList[dataListIndex] = Integer.parseInt(tokens.nextToken())-1;
				++dataListIndex;
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
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

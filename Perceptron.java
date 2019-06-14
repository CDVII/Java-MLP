/**
 * MLP에 사용되는 Perceptron.
 * MLP의 중간층과 출력층에 사용된다.
 * @author Hwiyong Chang
 */
public class Perceptron {
	private double threshold;
	private double[] weight; 
	
	Perceptron(int weightCount) {
		this.weight = new double[weightCount];
	}
	
	/**
	 * 임계값 설정
	 * @param value
	 */
	void setThreshold(double value) {
		this.threshold = value;
	}
	
	/**
	 * 임계값에 값 추가
	 * @param add
	 */
	void addThreshold(double add) {
		this.threshold += add;
	}				
	
	/**
	 * 임계값 반환
	 * @return
	 */
	double getThreshold() {
		return this.threshold;
	}
	
	/**
	 * 가중치값 설정
	 * @param index
	 * @param value
	 */
	void setWeight(int index, double value) {
		this.weight[index] = value;
	}
	
	/**
	 * 가중치값에 값 추가
	 * @param index
	 * @param add
	 */
	void addWeight(int index, double add) {
		this.weight[index] += add;
	}
	
	/**
	 * 가중치값 반환
	 * @param index
	 * @return
	 */
	double getWeight(int index) {
		return this.weight[index];
	}
}


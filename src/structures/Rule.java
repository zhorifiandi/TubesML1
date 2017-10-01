package structures;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;


public class Rule implements Serializable{
	private List<Edge> preconditions;
	private double class_value;
	
	public Rule(){
		preconditions = new ArrayList<Edge>();
		setClass_value(0);
	}

	public List<Edge> getPreconditions(){
		return preconditions;
	}

	public double getClass_value() {
		return class_value;
	}

	public void setClass_value(double class_value) {
		this.class_value = class_value;
	}

	public void addPrecondition(Edge att){
		preconditions.add(att);
	}
	
	public void delPrecondition(Edge att){
		preconditions.remove(att);
	}
	
	
}

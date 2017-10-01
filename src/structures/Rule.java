package structures;

import java.util.ArrayList;
import java.util.List;

import weka.core.Attribute;

public class Rule {
	private List<Attribute> preconditions;
	private double class_value;
	
	public Rule(){
		preconditions = new ArrayList<Attribute>();
		setClass_value(0);
	}

	public double getClass_value() {
		return class_value;
	}

	public void setClass_value(double class_value) {
		this.class_value = class_value;
	}

	public void addPrecondition(Attribute att){
		preconditions.add(att);
	}
	
	public void delPrecondition(Attribute att){
		preconditions.remove(att);
	}
	
	
}

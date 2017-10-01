package structures;

import java.io.Serializable;

/**
 * Created by zhorifiandi on 10/1/17.
 */
public class Edge implements Serializable{
    private String attribute_name;
    private int attribute_index;
    private String attribute_value;
    private double attribute_valueDouble;

    public String getAttribute_name() {
        return attribute_name;
    }

    public String getAttribute_value() {
        return attribute_value;
    }

    public void setAttribute_value(String attribute_value) {
        this.attribute_value = attribute_value;
    }

    public void setAttribute_name(String attribute_name) {
        this.attribute_name = attribute_name;
    }

    public int getAttribute_index() {
        return attribute_index;
    }

    public void setAttribute_index(int attribute_index) {
        this.attribute_index = attribute_index;
    }

    public double getAttribute_valueDouble() {
        return attribute_valueDouble;
    }

    public void setAttribute_valueDouble(double attribute_valueDouble) {
        this.attribute_valueDouble = attribute_valueDouble;
    }
}

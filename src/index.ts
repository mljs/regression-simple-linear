import type { NumberArray } from 'cheminfo-types';
import {
  BaseRegression,
  checkArrayLength,
  maybeToPrecision,
} from 'ml-regression-base';

type JsonType = ReturnType<SimpleLinearRegression['toJSON']>;

/**
 * Class representing simple linear regression.
 * The regression uses OLS to calculate intercept and slope.
 */
export class SimpleLinearRegression extends BaseRegression {
  slope: number;
  intercept: number;
  coefficients: number[];

  /**
   * @param x - explanatory variable
   * @param y - response variable
   */
  constructor(x: NumberArray, y: NumberArray) {
    super();
    // @ts-expect-error internal use of the constructor, from `this.load`
    if (x === true) {
      // @ts-expect-error internal use of the constructor, from `this.load`
      const yObj = y as JsonType;
      this.slope = yObj.slope;
      this.intercept = yObj.intercept;
      this.coefficients = [yObj.intercept, yObj.slope];
    } else {
      checkArrayLength(x, y);
      const result = regress(x, y);
      this.slope = result.slope;
      this.intercept = result.intercept;
      this.coefficients = [result.intercept, result.slope];
    }
  }

  /**
   * Get the parameters and model name in JSON format
   * @returns
   */
  toJSON() {
    return {
      name: 'simpleLinearRegression',
      slope: this.slope,
      intercept: this.intercept,
    };
  }

  _predict(x: number): number {
    return this.slope * x + this.intercept;
  }
  /**
   * Finds x for the given y value.
   * @param y - response variable value
   * @returns - x value
   */
  computeX(y: number): number {
    return (y - this.intercept) / this.slope;
  }

  /**
   * Strings the linear function in the form 'f(x) = ax + b'
   * @param precision - number of significant figures.
   * @returns
   */
  toString(precision?: number): string {
    let result = 'f(x) = ';
    if (this.slope !== 0) {
      const xFactor = maybeToPrecision(this.slope, precision);
      result += `${xFactor === '1' ? '' : `${xFactor} * `}x`;
      if (this.intercept !== 0) {
        const absIntercept = Math.abs(this.intercept);
        const operator = absIntercept === this.intercept ? '+' : '-';
        result += ` ${operator} ${maybeToPrecision(absIntercept, precision)}`;
      }
    } else {
      result += maybeToPrecision(this.intercept, precision);
    }
    return result;
  }
  /**
   * Strings the linear function in the form 'f(x) = ax + b'
   * @param precision - number of significant figures.
   * @returns
   */
  toLaTeX(precision?: number): string {
    return this.toString(precision);
  }

  /**
   * Class instance from a JSON Object.
   * @param json
   * @returns
   */
  static load(json: JsonType): SimpleLinearRegression {
    if (json.name !== 'simpleLinearRegression') {
      throw new TypeError('not a SLR model');
    }
    // @ts-expect-error internal use of the constructor
    return new SimpleLinearRegression(true, json);
  }
}

/**
 * Internal  function.
 * It determines the parameters (slope, intercept) of the line that best fit the `x,y` vector-data (simple linear regression).
 * @param x - explanatory variable
 * @param y - response variable
 * @returns - slope and intercept of the best fit line
 */
function regress(x: NumberArray, y: NumberArray) {
  const n = x.length;
  let xSum = 0;
  let ySum = 0;

  let xSquared = 0;
  let xY = 0;

  for (let i = 0; i < n; i++) {
    xSum += x[i];
    ySum += y[i];
    xSquared += x[i] * x[i];
    xY += x[i] * y[i];
  }

  const numerator = n * xY - xSum * ySum;

  const slope = numerator / (n * xSquared - xSum * xSum);
  return {
    slope,
    intercept: (1 / n) * ySum - slope * (1 / n) * xSum,
  };
}

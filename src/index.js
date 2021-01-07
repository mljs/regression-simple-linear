import BaseRegression, {
  checkArrayLength,
  maybeToPrecision,
} from 'ml-regression-base';

export default class SimpleLinearRegression extends BaseRegression {
  constructor(x, y) {
    super();
    if (x === true) {
      this.slope = y.slope;
      this.intercept = y.intercept;
      this.coefficients = [y.intercept, y.slope];
    } else {
      checkArrayLength(x, y);
      regress(this, x, y);
    }
  }

  toJSON() {
    return {
      name: 'simpleLinearRegression',
      slope: this.slope,
      intercept: this.intercept,
    };
  }

  _predict(x) {
    return this.slope * x + this.intercept;
  }

  computeX(y) {
    return (y - this.intercept) / this.slope;
  }

  toString(precision) {
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

  toLaTeX(precision) {
    return this.toString(precision);
  }

  static load(json) {
    if (json.name !== 'simpleLinearRegression') {
      throw new TypeError('not a SLR model');
    }
    return new SimpleLinearRegression(true, json);
  }
}

function regress(slr, x, y) {
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
  slr.slope = numerator / (n * xSquared - xSum * xSum);
  slr.intercept = (1 / n) * ySum - slr.slope * (1 / n) * xSum;
  slr.coefficients = [slr.intercept, slr.slope];
}

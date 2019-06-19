import BaseRegression from 'ml-regression-base';

declare module 'ml-regression-simple-linear' {
  export interface SLRModel {
    name: 'simpleLinearRegression';
  }

  class SimpleLinearRegression extends BaseRegression {
    slope: number;
    intercept: number;
    coefficients: [number, number];

    constructor(x: number[], y: number[]);

    static load(model: SLRModel): SimpleLinearRegression;

    computeX(y: number): number;
    toJSON(): SLRModel;
  }

  export = SimpleLinearRegression;
}

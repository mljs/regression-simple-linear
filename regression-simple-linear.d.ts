import BaseRegression from 'ml-regression-base';

declare module 'ml-regression-simple-linea' {
  export interface SLRModel {
    name: 'simpleLinearRegression';
  }

  export default class SimpleLinearRegression extends BaseRegression {
    constructor(x: number[], y: number[]);

    static load(model: SLRModel): SimpleLinearRegression;

    computeX(y: number): number;
    toJSON(): SLRModel;
  }
}

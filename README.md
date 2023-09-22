# regression-simple-linear

[![NPM version][npm-image]][npm-url]
[![build status][ci-image]][ci-url]
[![npm download][download-image]][download-url]
[![codecov][codecov-image]][codecov-url]

Simple Linear Regression.

## Installation

`$ npm install --save ml-regression-simple-linear`

## Usage

```js
import { SimpleLinearRegression } from 'ml-regression-simple-linear';

const x = [0.5, 1, 1.5, 2, 2.5];
const y = [0, 1, 2, 3, 4];

const regression = new SimpleLinearRegression(x, y);

regression.slope; // 2
regression.intercept; // -1
regression.coefficients; // [-1, 2]

regression.predict(3); // 5
regression.computeX(3.5); // 2.25

regression.toString(); // 'f(x) = 2 * x - 1'

regression.score(x, y);
// { r: 1, r2: 1, chi2: 0, rmsd: 0 }

const json = regression.toJSON();
// { name: 'simpleLinearRegression', slope: 2, intercept: -1 }
const loaded = SimpleLinearRegression.load(json);
loaded.predict(5); // 9
```

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-regression-simple-linear.svg?style=flat-square
[npm-url]: https://npmjs.org/package/ml-regression-simple-linear
[ci-image]: https://github.com/mljs/regression-simple-linear/workflows/Node.js%20CI/badge.svg?branch=master
[ci-url]: https://github.com/mljs/regression-simple-linear/actions?query=workflow%3A%22Node.js+CI%22
[download-image]: https://img.shields.io/npm/dm/ml-regression-simple-linear.svg?style=flat-square
[download-url]: https://npmjs.org/package/ml-regression-simple-linear
[codecov-image]: https://img.shields.io/codecov/c/github/mljs/regression-simple-linear.svg
[codecov-url]: https://codecov.io/gh/mljs/regression-simple-linear

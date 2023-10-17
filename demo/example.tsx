import React from 'react';
import { PolynomialRegression } from 'ml-regression-polynomial';

import { SimpleLinearRegression } from '../src/index';
import {
  Plot,
  LineSeries,
  Axis,
  Legend,
  Heading,
  SeriesPoint,
} from 'react-plot';

const factor = 1;
const data = [
  { x: 0, y: 0 },
  { x: 1, y: 1 },
  { x: 2, y: 2 },
  { x: 3, y: 3 },
  { x: 4, y: 3 },
  { x: 5, y: 3 },
];

const x = data.map((d) => d.x);
const y = data.map((d) => d.y);

const calculations: SeriesPoint[][] = [];

for (let i = 0; i < 5; i++) {
  if (i === 0) {
    const r = new SimpleLinearRegression(x, y).predict(x);
    const plot = r.map((y, i) => ({ x: x[i], y }));
    calculations.push(plot);
  } else {
    const r = new PolynomialRegression(x, y, i, {
      interceptAtZero: i % 2 === 1,
    }).predict(x);
    const plot = r.map((y, i) => ({ x: x[i], y }));
    calculations.push(plot);
  }
}

export const Example = () => (
  <Plot
    width={1000}
    height={1000}
    margin={{ bottom: 50, left: 90, top: 50, right: 100 }}
  >
    <Heading
      title="Electrical characterization"
      subtitle="Current vs Voltage"
    />
    <LineSeries
      data={data}
      xAxis="x"
      yAxis="y"
      lineStyle={{ strokeWidth: 3 }}
      label="Raw Data"
      displayMarkers={false}
    />
    {calculations.map((plot, i) => (
      <LineSeries
        key={i}
        data={plot}
        xAxis="x"
        yAxis="y"
        label={
          i === 0
            ? 'Simple Linear Regression'
            : `Polynomial Regression (degree ${i})`
        }
      />
    ))}
    <Axis
      id="x"
      position="bottom"
      label="Drain voltage [V]"
      displayPrimaryGridLines
      max={6.1 / factor}
    />
    <Axis
      id="y"
      position="left"
      label="Drain current [mA]"
      displayPrimaryGridLines
      max={6.1 * factor}
    />
    <Legend position="right" />
  </Plot>
);

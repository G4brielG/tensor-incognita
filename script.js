// y=2x +3
const modelo = tf.sequential();

modelo.add(tf.layers.dense({
  inputShape: 1,
  units: 1,
}));
modelo.add(tf.layers.dense({
  units: 1,
}))

modelo.compile({
  optimizer: "sgd",
  loss: "meanSquaredError",
});

const xs = tf.tensor([1, 3, 5], [3, 1]);
const ys = tf.tensor([2, 6, 10], [3, 1]);

modelo
  .fit(xs, ys, { epochs: 100 })
  .then(() => {
    const TensorX = tf.tensor([2, 4, 6]);
    const datosTensor = modelo.predict(TensorX).dataSync();
    const [...valores] = datosTensor;

    const res = valores.map((y) => ({ y }));

    console.log("Resultados", res)

    tf.dispose([xs, ys, modelo, datosTensor, TensorX]);
  });
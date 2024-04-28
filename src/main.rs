use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{loss, Linear, Module, Optimizer, VarBuilder, VarMap};
use std::vec;

struct Model {
    ln1: Linear,
    ln2: Linear,
}

#[derive(Clone)]
pub struct Dataset {
    pub training_x: Tensor,
    pub training_y: Tensor,
    pub testing_x: Tensor,
    pub testing_y: Tensor,
}

impl Dataset {
    pub fn new(device: &Device) -> Result<Self> {
        let training_x_vector: Vec<Vec<f32>> = vec![vec![16., 4.], vec![1., 4.], vec![16., 3.]];
        let training_x = Tensor::new(training_x_vector.clone(), &device)?;
        let testing_x_vector: Vec<Vec<f32>> = vec![vec![1., 3.]];
        let testing_x = Tensor::new(testing_x_vector.clone(), &device)?;

        let training_y_vector: Vec<Vec<f32>> = vec![vec![98.], vec![81.], vec![26.]];
        let training_y = Tensor::new(training_y_vector.clone(), &device)?;

        let testing_y_vector: Vec<Vec<f32>> = vec![vec![13.]];
        let testing_y = Tensor::new(testing_y_vector.clone(), &device)?;

        Ok(Self {
            training_x,
            training_y,
            testing_x,
            testing_y,
        })
    }
}

impl Model {
    fn new(vs: VarBuilder, input_size: usize, output_size: usize) -> Result<Self> {
        let ln1 = candle_nn::linear(input_size, 2, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(2, output_size, vs.pp("ln2"))?;

        Ok(Self { ln1, ln2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.ln1.forward(x)?;
        self.ln2.forward(&x)
    }
}

fn train(data: Dataset, device: &Device) -> Result<Model> {
    let training_x = data.training_x.to_device(device)?;
    let training_y = data.training_y.to_device(device)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);

    let model = Model::new(vs.clone(), 2, 1)?;

    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), 0.000001)?;

    for epoch in 1..=1000 {
        let pred = model.forward(&training_x)?;
        let loss = loss::mse(&pred, &training_y)?;
        sgd.backward_step(&loss)?;

        println!(
            "Epoch: {epoch:3} Train loss: {:.6}",
            loss.to_scalar::<f32>()?,
        );
        if loss.to_scalar::<f32>()? <= 0.001 {
            break;
        }
    }

    Ok(model)
}

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;

    let dataset = Dataset::new(&device)?;

    let trained_model: Model;
    loop {
        println!("Trying to train neural network.");
        match train(dataset.clone(), &device) {
            Ok(model) => {
                trained_model = model;
                break;
            }
            Err(e) => Err(e)?,
        }
    }

    let testing_data = dataset.clone();
    let final_result = trained_model.forward(&testing_data.testing_x)?;

    let y_true = testing_data.testing_y.get(0)?.to_vec1::<f32>()?[0];
    let y_pred = final_result.get(0)?.to_vec1::<f32>()?[0];

    println!("y_true: {:?}", y_true as i32);
    println!("y_pred: {:?}", y_pred as i32);

    Ok(())
}

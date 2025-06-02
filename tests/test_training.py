from model_training import train_model

def test_train_model_output():
    resumes = ["Python developer", "Java engineer"]
    labels = [1, 0]

    model = train_model.train_model(resumes, labels)
    pred = model.predict(["Experienced in Java and Spring"])
    
    assert hasattr(pred, '__iter__')  # iterable result

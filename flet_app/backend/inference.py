import openvino.inference_engine as ie

class InferenceModel:
    def __init__(self, model_xml, model_bin):
        self.model_xml = model_xml
        self.model_bin = model_bin
        self.ie_core = ie.IECore()
        self.net = self.ie_core.read_network(model=self.model_xml, weights=self.model_bin)
        self.exec_net = self.ie_core.load_network(network=self.net, device_name="CPU")

    def infer(self, input_frame):
        input_blob = next(iter(self.net.input_info))
        output_blob = next(iter(self.net.outputs))
        
        # Preprocess the input frame if necessary
        self.exec_net.start_async(request_id=0, inputs={input_blob: input_frame})
        if self.exec_net.wait(request_id=0) == 0:
            return self.exec_net.outputs[output_blob]
        else:
            raise RuntimeError("Inference failed")
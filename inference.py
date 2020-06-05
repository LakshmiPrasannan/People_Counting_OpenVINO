#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None
        
    def load_model(self, model, CPU_EXTENSION, DEVICE, console_output = False):
        ### TODO: Load the model ###
        #print("Entered load_model function")
        model_xml = model
        self.plugin = IECore()
        ### Load IR files into their related class
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.network = IENetwork(model = model_xml, weights = model_bin)
        ### Add a CPU extension, if applicable.
        if not all_layers_supported(self.plugin, self.network, console_output = console_output):
            self.plugin.add_extension(CPU_EXTENSION, "CPU")
        
        ### Get the supported layers of the network
        #supported_layers = self.plugin.query_network(network = self.exec_network, device_name="CPU")
        ### TODO: Check for supported layers ###
        #unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        #if len(unsupported_layers) != 0:
            #print("Unsupported layers found:{}".format(unsupported_layers))
            #print("Check whether extensions are available to add to IECore.")
            #exit(1)
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        #print("Starting to load the network model to IR")
        self.exec_network = self.plugin.load_network(self.network, DEVICE)
        #print("IR successfully loaded into Inference Engine.")
        
        #Get the input layers also:
        self.input_blob = next(iter(self.network.inputs))
        #Declaring the output layer also:
        self.output_blob = next(iter(self.network.outputs))
        ### Note: You may need to update the function parameters. ###
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        input_shapes = {}
        for i in self.network.inputs:
            input_shapes[i] = (self.network.inputs[i].shape)
        return input_shapes
    def exec_net(self, net_input, request_id):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        self.infer_request_handle = self.exec_network.start_async(request_id, inputs = net_input)
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        status = self.infer_request_handle.wait()
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        out = self.infer_request_handle.outputs[self.output_blob]
        ### Note: You may need to update the function parameters. ###
        return out
def all_layers_supported(engine, network, console_output=False):
    ### TODO check if all layers are supported
    ### return True if all supported, False otherwise
    layers_supported = engine.query_network(network, device_name='CPU')
    layers = network.layers.keys()

    all_supported = True
    for l in layers:
        if l not in layers_supported:
            all_supported = False

    return all_supported
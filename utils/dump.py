import onnx
import numpy as np
import copy

ACCELERATED_NODES = ("MatMul", "Gemm", "Conv", "ConvTranspose")
OUT_DTYPE_NODES = ("MatMul", "Gemm", "Conv", "ConvTranspose")

def onnx_dtype_from_np_dtype(np_dtype):
    return onnx.helper.np_dtype_to_tensor_dtype(np_dtype)

def onnx_update_weight_to(onnx_model, weights_name, new_weights_name, dtype):
    for init__ in onnx_model.graph.initializer:
        if init__.name == weights_name:
            init__.name = new_weights_name
            init__.raw_data = (
                np.frombuffer(
                    init__.raw_data,
                    onnx.helper.tensor_dtype_to_np_dtype(
                        init__.data_type
                    ),
                )
                .astype(dtype)
                .tobytes()
            )
            init__.data_type = onnx_dtype_from_np_dtype(dtype)
            return True
    return False

def onnx_update_inputs_to(onnx_model, dtype):
    for input__ in onnx_model.graph.input:
        input__.type.tensor_type.elem_type = onnx_dtype_from_np_dtype(dtype)

def onnx_update_outputs_to(onnx_model, dtype):
    for output__ in onnx_model.graph.output:
        output__.type.tensor_type.elem_type = onnx_dtype_from_np_dtype(dtype)

def onnx_update_node_attributes(node):
    original_attributes = [attr__ for attr__ in node.attribute]
    for attribute in original_attributes:
        # Plinio-MATCH exporter add some attributes that are not standard-ONNX so they should be removed
        if not ("_" in attribute.name and attribute.name.split("_")[1]=="bits"):
            node.attribute.append(attribute)
        node.attribute.pop(0)

def add_cast_as(onnx_model, new_nodes, node_idx, node, cast_type, cast_name: str="cast"):
    new_nodes.append(
        onnx.helper.make_node(
            name=cast_name,
            op_type="Cast",
            inputs=[node.output[0]],
            outputs=[node.output[0] + "_" + cast_name],
            to=onnx.helper.np_dtype_to_tensor_dtype(cast_type),
        ),
    )
    for other_node in onnx_model.graph.node[node_idx+1:]:
        for i in range(len(other_node.input)):
            if other_node.input[i] == node.output[0]:
                other_node.input[i] = node.output[0] + "_" + cast_name

def onnx_update_accelerated_node(onnx_model, new_nodes, node_idx, node, dtype):
    # it may that the input is detached due to a bug about naming in Plinio-MATCH exporter
    integerize = np.issubdtype(dtype, np.integer)
    weights_dtype = dtype if not integerize else dtype if dtype.kind == 'i' else np.dtype('i' + str(dtype.itemsize))
    # if integerize:
    #     node.op_type = node.op_type+"Integer"
    original_inputs = [str(inp__) for inp__ in node.input]
    for input__ in original_inputs:
        new_input = None
        if "::" in input__:
            new_input_name = "_".join(input__.split("::"))
        else:
            new_input_name = input__
        onnx_update_weight_to(onnx_model, input__, new_input_name, weights_dtype)
        node.input.pop(0)
        node.input.append(new_input_name)
    new_nodes.append(node)
    if dtype.itemsize!=4:
        cast_name = "accumulator"
        if node.op_type in OUT_DTYPE_NODES:
            cast_name = "FAKE_CAST_TVM_OUT_DTYPE"
        add_cast_as(onnx_model, new_nodes, node_idx, node, np.dtype("int32" if integerize else "float32"), cast_name)

def onnx_update_constant_to(node, dtype):
    # read input and reinterpret it as int32
    node.attribute[0].t.raw_data = (
        np.frombuffer(
            node.attribute[0].t.raw_data,
            onnx.helper.tensor_dtype_to_np_dtype(
                node.attribute[0].t.data_type
            ),
        )
        .astype(dtype)
        .tobytes()
    )
    node.attribute[0].t.data_type = onnx_dtype_from_np_dtype(dtype)

def onnx_update_nodes(onnx_model, dtype):
    new_nodes = list()
    floating = np.issubdtype(dtype, np.floating)
    single_prec_constant_dtype = np.dtype("float32") if floating else np.dtype("int32")

    for node_idx,node in enumerate(onnx_model.graph.node):
        new_node = copy.deepcopy(node)
        onnx_update_node_attributes(new_node)
        if node.op_type in ACCELERATED_NODES:
            onnx_update_accelerated_node(onnx_model, new_nodes, node_idx, new_node, dtype)
        elif node.op_type == "Constant":
            onnx_update_constant_to(new_node, single_prec_constant_dtype)
            new_nodes.append(new_node)
        elif node.op_type == "Clip":
            new_nodes.append(new_node)
            add_cast_as(onnx_model, new_nodes, node_idx, node, dtype)
        else:
            new_nodes.append(new_node)
    for node_idx in range(len(onnx_model.graph.node)):
        # remove the original node
        onnx_model.graph.node.pop(0)
    onnx_model.graph.node.extend(new_nodes)

def sanitize_model_to_int32(model_path, new_model_path):
    onnx_model = onnx.load(model_path)
    
    onnx_update_inputs_to(onnx_model, np.dtype("int32"))
    onnx_update_outputs_to(onnx_model, np.dtype("int32"))
    onnx_update_nodes(onnx_model, np.dtype("int32"))
    
    with open(new_model_path, "wb") as new_f:
        onnx.save(onnx_model,new_f)

def sanitize_model_to_uint8(model_path, new_model_path):
    onnx_model = onnx.load(model_path)
    
    onnx_update_inputs_to(onnx_model, np.dtype("uint8"))
    onnx_update_outputs_to(onnx_model, np.dtype("int32"))
    onnx_update_nodes(onnx_model, np.dtype("uint8"))
    
    with open(new_model_path, "wb") as new_f:
        onnx.save(onnx_model,new_f)
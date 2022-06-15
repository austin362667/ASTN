package main

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
)

// Value stores a single scalar value and its gradient
type Value struct {
	data      float64
	grad      float64
	_backward func()
	_prev     map[string]Value
	_op       string
}

func (self Value) SetBackward(fn func()) Value {
	self._backward = fn
	return self
}

//func addBackward(self, other, out Value) {
//	self.grad += out.grad
//	other.grad += out.grad
//}

func (self Value) Add(other Value) Value {
	out := Value{self.data + other.data, 0, nil, map[string]Value{"self": self, "other": other}, "+"}
	out = out.SetBackward(func() {
		self.grad += out.grad
		other.grad += out.grad
	})
	return out
}

//func mulBackward(self, other, out Value) {
//	self.grad += other.grad * out.grad
//	other.grad += self.grad * out.grad
//}

func (self Value) Mul(other Value) Value {
	out := Value{self.data * other.data, 0, nil, map[string]Value{"self": self, "other": other}, "*"}
	out = out.SetBackward(func() {
		self.grad += other.grad * out.grad
		other.grad += self.grad * out.grad
	})
	return out
}

//func powBackward(self, other, out Value) {
//	// self.grad += (other * self.data**(other-1)) * out.grad
//	self.grad += other.data * math.Pow(self.data, other.data-1)
//}

func (self Value) Pow(other Value) Value {
	out := Value{math.Pow(self.data, other.data), 0, nil, map[string]Value{"self": self, "other": other}, "**"}
	out.SetBackward(func() {
		self.grad += other.data * math.Pow(self.data, other.data-1)
	})
	return out
}

//func reluBackward(self, out Value) {
//	// self.grad += (out.data > 0) * out.grad
//	od := 0.
//	if out.data > 0 {
//		od = out.data
//	} else {
//		od = 0.
//	}
//	self.grad += od * out.grad
//}

func (self Value) Relu() Value {
	sd := 0.
	if self.data > 0 {
		sd = self.data
	} else {
		sd = 0.
	}
	out := Value{sd, 0, nil, map[string]Value{"self": self}, "ReLU"}
	out.SetBackward(func() {
		od := 0.
		if out.data > 0 {
			od = out.data
		} else {
			od = 0.
		}
		self.grad += od * out.grad
	})
	return out
}

func build_topo(topo []Value, v Value, visited map[string]Value) {
	if v, ok := visited[Hash(v)]; !ok {
		visited[Hash(v)] = v
	}
	for _, child := range v._prev {
		build_topo(topo, child, visited)
	}
	topo = append(topo, v)
}

func Hash(s Value) string {
	var b bytes.Buffer
	gob.NewEncoder(&b).Encode(s)
	return b.String()
}

func (self Value) backward() {
	// topological order all of the children in the graph
	var topo []Value
	visited := make(map[string]Value)

	build_topo(topo, self, visited)

	// go one variable at a time and apply the chain rule to get its gradient
	self.grad = 1
	l := len(topo)
	for i := l - 1; i > 0; i-- {
		topo[i]._backward()
	}

}

func main() {
	va := Value{3, 0, nil, nil, ""}
	vb := Value{4, 0, nil, nil, ""}
	vc := Value{-1, 0, nil, nil, ""}
	fmt.Println(va.Add(vb).data)
	fmt.Println(va.Mul(vb).data)
	fmt.Println(va.Pow(vb).data)
	fmt.Println(vb.Relu().data)
	fmt.Println(vc.Relu().data)

	fmt.Println(vb.Relu().grad)
	vb.backward()
	fmt.Println(vb.data)

	x := Value{-4., 0., nil, nil, ""}
	two := Value{2., 0., nil, nil, ""}
	z := two.Mul(x).Add(two).Add(x)
	q := z.Relu().Add(z.Mul(x))
	h := z.Mul(z).Relu()
	y := h.Add(q).Add(q.Mul(x))
	z.backward()
	fmt.Println(x.grad, y.data)
}

type Module struct {
	_zerograd   func()
	_parameters func()
}

type Neuron struct {
	w      []Value
	b      Value
	nonlin bool
}

func (N Neuron) init(nin []Value, nonlin bool) {
	for _, n := range nin {
		fmt.Println(n)
		N.w = append(N.w, Value{rand.Float64(), 0, nil, nil, ""})
	}
	N.b = Value{0, 0, nil, nil, ""}
	N.nonlin = nonlin
}

func (N Neuron) call(x Value) Value {
	act := Value{0, 0, nil, nil, ""}
	for _, wi := range N.w {
		act = act.Add(wi.Mul(x))
	}
	act = act.Add(N.b)

	if N.nonlin {
		return act.Relu()
	}
	return act
}

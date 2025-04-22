package main

import (
	"fmt"
	"math"
)

// Rectangle represents a geometric rectangle.
type Rectangle struct {
	Height float64
	Width  float64
}

// Area calculates the area of the rectangle.
func (r Rectangle) Area() float64 {
	return r.Height * r.Width
}

// Perimeter calculates the perimeter of the rectangle.
func (r Rectangle) Perimeter() float64 {
	return 2 * (r.Height + r.Width)
}

// SayHello prints a greeting message.
func SayHello(name string) {
	fmt.Printf("Hello, %s!\n", name)
}

// Multiply returns the product of two integers.
func Multiply(a, b int) int {
	return a * b
}

// PI is the mathematical constant π.
const PI = 3.1415926535

// CircleArea calculates the area of a circle given its radius.
func CircleArea(radius float64) float64 {
	return PI * math.Pow(radius, 2)
}

// main is the entry point of the program.
func main() {
	rect := Rectangle{Height: 10, Width: 5}
	fmt.Println("Area:", rect.Area())
	fmt.Println("Perimeter:", rect.Perimeter())

	SayHello("Alice")
	fmt.Println("Product:", Multiply(3, 4))
	fmt.Println("Circle Area:", CircleArea(7))
}

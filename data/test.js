// utils.js

// Preprocessor-like comment (for testing comment handling)
"use strict";

// Class declaration
class Rectangle {
  constructor(height, width) {
    this.height = height;
    this.width = width;
  }

  area() {
    return this.height * this.width;
  }
}

// Function declaration
function sayHello(name) {
  console.log(`Hello, ${name}!`);
}

// Arrow function
const multiply = (a, b) => a * b;

// Variable declaration
const PI = 3.14;

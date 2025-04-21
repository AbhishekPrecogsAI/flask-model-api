"use strict";

class NetworkService {
  constructor(url) {
    this.url = url;
  }

  async fetchData() {
    return await fetch(this.url);
  }
}

function init() {
  console.log("App initialized");
}

const config = {
  mode: "production",
  retry: true
};

const fetchStatus = async () => {
  try {
    let res = await fetch("/status");
    return res.ok;
  } catch {
    return false;
  }
};

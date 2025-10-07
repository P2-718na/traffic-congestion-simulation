__kernel void mandelbrot(__global long * set, long width, long height, long depth) {

  int2 pos = (int2)(get_global_id(0), get_global_id(1));
  int2 size = (int2)(width, height);

  // Scale x to -2.5 to 1
  float x0 = (float)pos.x / width;
  x0 = x0*3.5 - 2.5;

  // Scale y to -1 to 1
  float y0 = (float)pos.y / height;
  y0 = y0*3.0 - 1.5;


  float x = 0.0;
  float y = 0.0;

  uint i = 0;
  float d = 0.0;

  while (i < depth && d < 4.){
    float xtemp = x*x - y*y + x0;
    y = 2*x*y + y0;
    x = xtemp;

    d = x*x + y*y;
    i++;
  }

  set[pos.x + width  * pos.y] = i;
}
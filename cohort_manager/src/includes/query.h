#ifndef GUARD_query_h
#define GUARD_query_h

#include <string>

struct Alignment { int score, left, right; };
enum Move { UP, LEFT, DIAG };
Alignment alignScore(const std::string& query, const std::string& word);


#endif

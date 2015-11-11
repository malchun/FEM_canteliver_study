/* ---------------------------------------------------------------------
 * Author: Wolfgang Bangerth, University of Texas at Austin, 2000, 2004, 2005,
 * Timo Heister, 2013, malchun, 2015
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/grid/filtered_iterator.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <sys/types.h>
#include <sys/stat.h>

using namespace dealii;

template <int dim>
struct PointHistory {
  SymmetricTensor<2, dim> old_stress;
};

template <int dim>
SymmetricTensor<4, dim> get_stress_strain_tensor(const double lambda,
                                                 const double mu) {
  SymmetricTensor<4, dim> tmp;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      for (unsigned int k = 0; k < dim; ++k)
        for (unsigned int l = 0; l < dim; ++l)
          tmp[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) +
                             ((i == l) && (j == k) ? mu : 0.0) +
                             ((i == j) && (k == l) ? lambda : 0.0));
  return tmp;
}

template <int dim>
inline SymmetricTensor<2, dim> get_strain(const FEValues<dim> &fe_values,
                                          const unsigned int shape_func,
                                          const unsigned int q_point) {
  SymmetricTensor<2, dim> tmp;

  for (unsigned int i = 0; i < dim; ++i)
    tmp[i][i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];

  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = i + 1; j < dim; ++j)
      tmp[i][j] = (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
                   fe_values.shape_grad_component(shape_func, q_point, j)[i]) /
                  2;

  return tmp;
}

template <int dim>
inline SymmetricTensor<2, dim> get_strain(
    const std::vector<Tensor<1, dim>> &grad) {
  Assert(grad.size() == dim, ExcInternalError());

  SymmetricTensor<2, dim> strain;
  for (unsigned int i = 0; i < dim; ++i) strain[i][i] = grad[i][i];

  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = i + 1; j < dim; ++j)
      strain[i][j] = (grad[i][j] + grad[j][i]) / 2;

  return strain;
}

Tensor<2, 2> get_rotation_matrix(const std::vector<Tensor<1, 2>> &grad_u) {
  const double curl = (grad_u[1][0] - grad_u[0][1]);

  const double angle = std::atan(curl);

  const double t[2][2] = {{cos(angle), sin(angle)}, {-sin(angle), cos(angle)}};
  return Tensor<2, 2>(t);
}

Tensor<2, 3> get_rotation_matrix(const std::vector<Tensor<1, 3>> &grad_u) {
  const Point<3> curl(grad_u[2][1] - grad_u[1][2], grad_u[0][2] - grad_u[2][0],
                      grad_u[1][0] - grad_u[0][1]);

  const double tan_angle = std::sqrt(curl * curl);
  const double angle = std::atan(tan_angle);

  if (angle < 1e-9) {
    static const double rotation[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    static const Tensor<2, 3> rot(rotation);
    return rot;
  }

  const double c = std::cos(angle);
  const double s = std::sin(angle);
  const double t = 1 - c;

  const Point<3> axis = curl / tan_angle;
  const double rotation[3][3] = {
      {t * axis[0] * axis[0] + c, t * axis[0] * axis[1] + s * axis[2],
       t * axis[0] * axis[2] - s * axis[1]},
      {t * axis[0] * axis[1] - s * axis[2], t * axis[1] * axis[1] + c,
       t * axis[1] * axis[2] + s * axis[0]},
      {t * axis[0] * axis[2] + s * axis[1], t * axis[1] * axis[1] - s * axis[0],
       t * axis[2] * axis[2] + c}};
  return Tensor<2, 3>(rotation);
}


// Главный черномагический класс. Делает почти все. Крайне неудобен, требует 
// переписывания с нуля.

// Если поконкретнее - создает сетку, систему, запускает решение, организует вывод.
// Была ещё функция уточнения сетки на основе первого шага, но за плохую работу
// выброшена (временно, идея интересная).
template <int dim>
class TopLevel {
 public:
  TopLevel();
  ~TopLevel();
  void run();

 private:
  void create_coarse_grid();

  void setup_system();

  void assemble_system();

  void solve_timestep();

  unsigned int solve_linear_problem();

  void output_results() const;

  void do_initial_timestep();

  void do_timestep();

  void refine_initial_grid();

  void move_mesh();

  void setup_quadrature_point_history();

  void update_quadrature_point_history();

  Triangulation<dim> triangulation;

  FESystem<dim> fe;

  DoFHandler<dim> dof_handler;

  ConstraintMatrix hanging_node_constraints;

  const QGauss<dim> quadrature_formula;

  std::vector<PointHistory<dim>> quadrature_point_history;

  PETScWrappers::MPI::SparseMatrix system_matrix;

  PETScWrappers::MPI::Vector system_rhs;

  Vector<double> incremental_displacement;

  double present_time;
  double present_timestep;
  double end_time;
  unsigned int timestep_no;

  MPI_Comm mpi_communicator;

  const unsigned int n_mpi_processes;

  const unsigned int this_mpi_process;

  ConditionalOStream pcout;

  std::vector<types::global_dof_index> local_dofs_per_process;

  types::global_dof_index n_local_dofs;

  unsigned int n_local_cells;

  static const SymmetricTensor<4, dim> stress_strain_tensor;
};



// Класс внутренних сил. По сути задает вектор для каждой точки (по необходимости).
// В данном примере - только тяжесть.
template <int dim>
class BodyForce : public Function<dim> {
 public:
  BodyForce();

  virtual void vector_value(const Point<dim> &p, Vector<double> &values) const;

  virtual void vector_value_list(const std::vector<Point<dim>> &points,
                                 std::vector<Vector<double>> &value_list) const;
};

template <int dim>
BodyForce<dim>::BodyForce()
    : Function<dim>(dim) {}

template <int dim>
inline void BodyForce<dim>::vector_value(const Point<dim> & p,
                                         Vector<double> &values) const {
  Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));

  //================ Values!!!!! ======================
  const double g = 9.81;
  const double rho = 7700;

  values = 0;
  // Немного тяжести по z 
  values(dim - 1) = -rho * g;
  // И по y тоже, но побольше.
  if ((9.5 < p[2]) &&
      (10.5 > p[2]) &&
      (-0.4 < p[0]) &&
      (0.4 > p[0])) {
    values(1) = rho * g * 50000;
  }
}

template <int dim>
void BodyForce<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &value_list) const {
  const unsigned int n_points = points.size();

  Assert(value_list.size() == n_points,
         ExcDimensionMismatch(value_list.size(), n_points));

  for (unsigned int p = 0; p < n_points; ++p)
    BodyForce<dim>::vector_value(points[p], value_list[p]);
}




// Класс внешних сил. Аналогично внутренним.
template <int dim>
class IncrementalBoundaryValues : public Function<dim> {
 public:
  IncrementalBoundaryValues(const double present_time,
                            const double present_timestep);

  virtual void vector_value(const Point<dim> &p, Vector<double> &values) const;

  virtual void vector_value_list(const std::vector<Point<dim>> &points,
                                 std::vector<Vector<double>> &value_list) const;

 private:
  const double velocity;
  const double present_time;
  const double present_timestep;
};

template <int dim>
IncrementalBoundaryValues<dim>::IncrementalBoundaryValues(
    const double present_time, const double present_timestep)
    : Function<dim>(dim),
      velocity(.1),
      present_time(present_time),
      present_timestep(present_timestep) {}

template <int dim>
void IncrementalBoundaryValues<dim>::vector_value(
    const Point<dim> & /*p*/, Vector<double> &values) const {
  Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));

  //================ Values!!!!! ======================
  // Здесь задается вектор внешней силы. Один. 
  // Не могу перевести в нормальные величины.
  // Upd Это не силы, это просто скорость.
  values = 0;
  // x
  //values(0) = velocity * 1;
  // y
  //values(1) = velocity * 1;
}

template <int dim>
void IncrementalBoundaryValues<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &value_list) const {
  const unsigned int n_points = points.size();

  Assert(value_list.size() == n_points,
         ExcDimensionMismatch(value_list.size(), n_points));

  for (unsigned int p = 0; p < n_points; ++p)
    IncrementalBoundaryValues<dim>::vector_value(points[p], value_list[p]);
}


// Вот и он - главный тензор!
template <int dim>
const SymmetricTensor<4, dim> TopLevel<dim>::stress_strain_tensor =
    get_stress_strain_tensor<dim>(/*lambda = */ 9.695e10,
                                  /*mu     = */ 7.617e10);



template <int dim>
TopLevel<dim>::TopLevel()
    : fe(FE_Q<dim>(1), dim),
      dof_handler(triangulation),
      quadrature_formula(2),
      mpi_communicator(MPI_COMM_WORLD),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
      pcout(std::cout, this_mpi_process == 0) {}

template <int dim>
TopLevel<dim>::~TopLevel() {
  dof_handler.clear();
}

// Собственно первая функция. Задает кол-во итераций и шаг симуляции.
template <int dim>
void TopLevel<dim>::run() {
  // ================ YOHOHOHO =================
  present_time = 0;
  present_timestep = 1;
  end_time = 10;
  timestep_no = 0;

  do_initial_timestep();

  while (present_time < end_time) do_timestep();
}

// ============= Totally mine! ================
// Название "грубая сетка" в данный момент не вполне корректно.
// Здесь здается сетка и отмечаются интересные участки поверхности.
template <int dim>
void TopLevel<dim>::create_coarse_grid() {
  Point<dim> p1 = {-1, -1, 0}, p2 = {1, 1, 20};
  const std::vector<unsigned int> repetitions{10, 10, 100};
  GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2,
                                            false);

  for (typename Triangulation<dim>::active_cell_iterator cell =
           triangulation.begin_active();
       cell != triangulation.end(); ++cell)
    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary()) {
        const Point<dim> face_center = cell->face(f)->center();

        if (0 == face_center[2])
          cell->face(f)->set_boundary_id(0);
        else if (20 == face_center[2])
          cell->face(f)->set_boundary_id(1);
        else if ((9.8 < face_center[2]) &&
                 (10.2 > face_center[2]) &&
                 (-0.2 < face_center[0]) &&
                 (0.2 > face_center[0]) &&
                 (-1 == face_center[1]))
          cell->face(f)->set_boundary_id(4);
        else
          cell->face(f)->set_boundary_id(2);
      }

  triangulation.refine_global(0);

  GridTools::partition_triangulation(n_mpi_processes, triangulation);
  setup_quadrature_point_history();
}


// Присваивание степеней свободы (магически генерирующихся), 
// инициализация всего что относится к счету. 
template <int dim>
void TopLevel<dim>::setup_system() {
  dof_handler.distribute_dofs(fe);
  DoFRenumbering::subdomain_wise(dof_handler);

  n_local_cells = GridTools::count_cells_with_subdomain_association(
      triangulation, this_mpi_process);

  local_dofs_per_process.resize(n_mpi_processes);
  for (unsigned int i = 0; i < n_mpi_processes; ++i)
    local_dofs_per_process[i] =
        DoFTools::count_dofs_with_subdomain_association(dof_handler, i);

  n_local_dofs = local_dofs_per_process[this_mpi_process];

  hanging_node_constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler,
                                          hanging_node_constraints);
  hanging_node_constraints.close();

  DynamicSparsityPattern sparsity_pattern(dof_handler.n_dofs(),
                                          dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern);
  hanging_node_constraints.condense(sparsity_pattern);
  system_matrix.reinit(mpi_communicator, sparsity_pattern,
                       local_dofs_per_process, local_dofs_per_process,
                       this_mpi_process);
  system_rhs.reinit(mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
  incremental_displacement.reinit(dof_handler.n_dofs());
}

// Здесь наконец-то и заводится FEM система.
template <int dim>
void TopLevel<dim>::assemble_system() {
  system_rhs = 0;
  system_matrix = 0;

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  BodyForce<dim> body_force;
  std::vector<Vector<double>> body_force_values(n_q_points,
                                                Vector<double>(dim));

  typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                 endc = dof_handler.end();
  for (; cell != endc; ++cell)
    if (cell->subdomain_id() == this_mpi_process) {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            const SymmetricTensor<2, dim> eps_phi_i =
                                              get_strain(fe_values, i, q_point),
                                          eps_phi_j =
                                              get_strain(fe_values, j, q_point);

            cell_matrix(i, j) += (eps_phi_i * stress_strain_tensor * eps_phi_j *
                                  fe_values.JxW(q_point));
          }

      const PointHistory<dim> *local_quadrature_points_data =
          reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
      body_force.vector_value_list(fe_values.get_quadrature_points(),
                                   body_force_values);

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        const unsigned int component_i = fe.system_to_component_index(i).first;

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
          const SymmetricTensor<2, dim> &old_stress =
              local_quadrature_points_data[q_point].old_stress;

          cell_rhs(i) += (body_force_values[q_point](
                              component_i)*fe_values.shape_value(i, q_point) -
                          old_stress * get_strain(fe_values, i, q_point)) *
                         fe_values.JxW(q_point);
        }
      }

      cell->get_dof_indices(local_dof_indices);

      hanging_node_constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // Указание областей воздействия сил. 
  // ========================== Values ===============================
  //FEValuesExtractors::Scalar z_component(dim - 1);
  //FEValuesExtractors::Scalar y_component(dim - 2);
  std::map<types::global_dof_index, double> boundary_values;
  // Симулируются концы в "зажимах"
  VectorTools::interpolate_boundary_values(
      dof_handler, 0, ZeroFunction<dim>(dim), boundary_values);
  VectorTools::interpolate_boundary_values(
      dof_handler, 1, ZeroFunction<dim>(dim), boundary_values);
/*  
  // Нагрузка. По желанию, можно явно указать маску.
  // Upd - нифига это не силы. Это заданное движение границы. 
  // Поэтому и не останавливается. Как только смогу - переделаю.
  VectorTools::interpolate_boundary_values(
      dof_handler, 4,
      IncrementalBoundaryValues<dim>(present_time, present_timestep),
      boundary_values);
  //                                 fe.component_mask(y_component));
*/
  PETScWrappers::MPI::Vector tmp(mpi_communicator, dof_handler.n_dofs(),
                                 n_local_dofs);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, tmp,
                                     system_rhs, false);
  incremental_displacement = tmp;
}

template <int dim>
void TopLevel<dim>::solve_timestep() {
  pcout << "    Assembling system..." << std::flush;
  assemble_system();
  pcout << " norm of rhs is " << system_rhs.l2_norm() << std::endl;

  const unsigned int n_iterations = solve_linear_problem();

  pcout << "    Solver converged in " << n_iterations << " iterations."
        << std::endl;

  pcout << "    Updating quadrature point data..." << std::flush;
  update_quadrature_point_history();
  pcout << std::endl;
}


template <int dim>
unsigned int TopLevel<dim>::solve_linear_problem() {
  PETScWrappers::MPI::Vector distributed_incremental_displacement(
      mpi_communicator, dof_handler.n_dofs(), n_local_dofs);
  distributed_incremental_displacement = incremental_displacement;

  SolverControl solver_control(dof_handler.n_dofs(),
                               1e-16 * system_rhs.l2_norm());
  PETScWrappers::SolverCG cg(solver_control, mpi_communicator);

  PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);

  cg.solve(system_matrix, distributed_incremental_displacement, system_rhs,
           preconditioner);

  incremental_displacement = distributed_incremental_displacement;

  hanging_node_constraints.distribute(incremental_displacement);

  return solver_control.last_step();
}


// Черезвычайно хитрый класс, который должен правильно организовать
// вывод многопроцессорного выполнения. Трогать боюсь. 
template <int dim>
class FilteredDataOut : public DataOut<dim> {
 public:
  FilteredDataOut(const unsigned int subdomain_id)
      : subdomain_id(subdomain_id) {}

  virtual typename DataOut<dim>::cell_iterator first_cell() {
    typename DataOut<dim>::active_cell_iterator cell =
        this->dofs->begin_active();
    while ((cell != this->dofs->end()) &&
           (cell->subdomain_id() != subdomain_id))
      ++cell;

    return cell;
  }

  virtual typename DataOut<dim>::cell_iterator next_cell(
      const typename DataOut<dim>::cell_iterator &old_cell) {
    if (old_cell != this->dofs->end()) {
      const IteratorFilters::SubdomainEqualTo predicate(subdomain_id);

      return ++(FilteredIterator<typename DataOut<dim>::active_cell_iterator>(
          predicate, old_cell));
    } else
      return old_cell;
  }

 private:
  const unsigned int subdomain_id;
};

// Вывод, который организуется через данный черезвычайно хитрый класс.
// Пишет в папку result.
template <int dim>
void TopLevel<dim>::output_results() const {
  FilteredDataOut<dim> data_out(this_mpi_process);
  data_out.attach_dof_handler(dof_handler);

  std::vector<std::string> solution_names;
  switch (dim) {
    case 1:
      solution_names.push_back("delta_x");
      break;
    case 2:
      solution_names.push_back("delta_x");
      solution_names.push_back("delta_y");
      break;
    case 3:
      solution_names.push_back("delta_x");
      solution_names.push_back("delta_y");
      solution_names.push_back("delta_z");
      break;
    default:
      Assert(false, ExcNotImplemented());
  }

  data_out.add_data_vector(incremental_displacement, solution_names);

  Vector<double> norm_of_stress(triangulation.n_active_cells());
  {
    typename Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();
    for (unsigned int index = 0; cell != endc; ++cell, ++index)
      if (cell->subdomain_id() == this_mpi_process) {
        SymmetricTensor<2, dim> accumulated_stress;
        for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
          accumulated_stress += reinterpret_cast<PointHistory<dim> *>(
                                    cell->user_pointer())[q].old_stress;

        norm_of_stress(index) =
            (accumulated_stress / quadrature_formula.size()).norm();
      } else
        norm_of_stress(index) = -1e+20;
  }
  data_out.add_data_vector(norm_of_stress, "norm_of_stress");

  std::vector<types::subdomain_id> partition_int(
      triangulation.n_active_cells());
  GridTools::get_subdomain_association(triangulation, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  std::string filename = "result/solution-" +
                         Utilities::int_to_string(timestep_no, 4) + "." +
                         Utilities::int_to_string(this_mpi_process, 3) + ".vtu";

  AssertThrow(n_mpi_processes < 1000, ExcNotImplemented());

  std::ofstream output(filename.c_str());
  data_out.write_vtu(output);

  if (this_mpi_process == 0) {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < n_mpi_processes; ++i)
      filenames.push_back("result/solution-" +
                          Utilities::int_to_string(timestep_no, 4) + "." +
                          Utilities::int_to_string(i, 3) + ".vtu");
  }
}


// Шаг, в котором создается сетка и система уравнений
template <int dim>
void TopLevel<dim>::do_initial_timestep() {
  present_time += present_timestep;
  ++timestep_no;
  pcout << "Timestep " << timestep_no << " at time " << present_time
        << std::endl;

  // ============== YOHOHOHO ==================
  create_coarse_grid();
  pcout << "    Number of active cells:       "
        << triangulation.n_active_cells() << " (by partition:";
  for (unsigned int p = 0; p < n_mpi_processes; ++p)
    pcout << (p == 0 ? ' ' : '+')
          << (GridTools::count_cells_with_subdomain_association(triangulation,
                                                                p));
  pcout << ")" << std::endl;

  setup_system();

  pcout << "    Number of degrees of freedom: " << dof_handler.n_dofs()
        << " (by partition:";
  for (unsigned int p = 0; p < n_mpi_processes; ++p)
    pcout << (p == 0 ? ' ' : '+')
          << (DoFTools::count_dofs_with_subdomain_association(dof_handler, p));
  pcout << ")" << std::endl;

  solve_timestep();

  move_mesh();
  output_results();

  pcout << std::endl;
}



// последующие шаги
template <int dim>
void TopLevel<dim>::do_timestep() {
  present_time += present_timestep;
  ++timestep_no;
  pcout << "Timestep " << timestep_no << " at time " << present_time
        << std::endl;
  if (present_time > end_time) {
    present_timestep -= (present_time - end_time);
    present_time = end_time;
  }

  solve_timestep();

  move_mesh();
  output_results();

  pcout << std::endl;
}

// Название говорящее.
template <int dim>
void TopLevel<dim>::move_mesh() {
  pcout << "    Moving mesh..." << std::endl;

  std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
  for (typename DoFHandler<dim>::active_cell_iterator cell =
           dof_handler.begin_active();
       cell != dof_handler.end(); ++cell)
    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
      if (vertex_touched[cell->vertex_index(v)] == false) {
        vertex_touched[cell->vertex_index(v)] = true;

        Point<dim> vertex_displacement;
        for (unsigned int d = 0; d < dim; ++d)
          vertex_displacement[d] =
              incremental_displacement(cell->vertex_dof_index(v, d));

        cell->vertex(v) += vertex_displacement;
      }
}

// ДОБАВИТЬ ОПИСАНИЕ
template <int dim>
void TopLevel<dim>::setup_quadrature_point_history() {
  unsigned int our_cells = 0;
  for (typename Triangulation<dim>::active_cell_iterator cell =
           triangulation.begin_active();
       cell != triangulation.end(); ++cell)
    if (cell->subdomain_id() == this_mpi_process) ++our_cells;

  triangulation.clear_user_data();

  {
    std::vector<PointHistory<dim>> tmp;
    tmp.swap(quadrature_point_history);
  }
  quadrature_point_history.resize(our_cells * quadrature_formula.size());

  unsigned int history_index = 0;
  for (typename Triangulation<dim>::active_cell_iterator cell =
           triangulation.begin_active();
       cell != triangulation.end(); ++cell)
    if (cell->subdomain_id() == this_mpi_process) {
      cell->set_user_pointer(&quadrature_point_history[history_index]);
      history_index += quadrature_formula.size();
    }

  Assert(history_index == quadrature_point_history.size(), ExcInternalError());
}

// ДОБАВИТЬ ОПИСАНИЕ
template <int dim>
void TopLevel<dim>::update_quadrature_point_history() {
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients);
  std::vector<std::vector<Tensor<1, dim>>> displacement_increment_grads(
      quadrature_formula.size(), std::vector<Tensor<1, dim>>(dim));

  for (typename DoFHandler<dim>::active_cell_iterator cell =
           dof_handler.begin_active();
       cell != dof_handler.end(); ++cell)
    if (cell->subdomain_id() == this_mpi_process) {
      PointHistory<dim> *local_quadrature_points_history =
          reinterpret_cast<PointHistory<dim> *>(cell->user_pointer());
      Assert(
          local_quadrature_points_history >= &quadrature_point_history.front(),
          ExcInternalError());
      Assert(local_quadrature_points_history < &quadrature_point_history.back(),
             ExcInternalError());

      fe_values.reinit(cell);
      fe_values.get_function_gradients(incremental_displacement,
                                       displacement_increment_grads);

      for (unsigned int q = 0; q < quadrature_formula.size(); ++q) {
        const SymmetricTensor<2, dim> new_stress =
            (local_quadrature_points_history[q].old_stress +
             (stress_strain_tensor *
              get_strain(displacement_increment_grads[q])));

        const Tensor<2, dim> rotation =
            get_rotation_matrix(displacement_increment_grads[q]);
        const SymmetricTensor<2, dim> rotated_new_stress =
            symmetrize(transpose(rotation) *
                       static_cast<Tensor<2, dim>>(new_stress) * rotation);
        local_quadrature_points_history[q].old_stress = rotated_new_stress;
      }
    }
}



int main (int argc, char **argv)
{
    struct stat sb;

    if (stat("result/", &sb) != 0 || !S_ISDIR(sb.st_mode)) {
        std::cout << "Can't find directory result/. Aborting." << std::endl;
        return 0;
    }
    try {
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        {
            deallog.depth_console (0);

            TopLevel<3> elastic_problem;
            elastic_problem.run ();
        }
    } catch (std::exception &exc) {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
    } catch (...) {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    return 0;
}
